import argparse
import copy
import json
import os
from os import path

import dask
import dask.dataframe as dd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from lpis_processing import config_s3 as config

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

key = config.key
secret = config.secret
endpoint_url = config.endpoint_url
bucket_name = config.bucket_name


def prepare_file(input_dir, aoi, year, icc_code):
    barycenters_file = f'{input_dir}/{aoi}/{year}/barycenter_poly_{icc_code}.parquet'
    barycenters_data = dd.read_parquet(barycenters_file, storage_options={
        "key": key,
        "secret": secret,
        "client_kwargs": {"endpoint_url": endpoint_url}
    })
    print(f'Prepare file {aoi} {year} {icc_code}')
    print(dask.compute(*barycenters_data.shape))
    barycenters_data_df = barycenters_data.compute()

    # compute nan vals per column
    nan_vals = barycenters_data_df.isna().sum()
    nan_vals_df = nan_vals.reset_index()

    sel_cols = list(nan_vals_df[nan_vals_df[0] <= (barycenters_data_df.shape[0] / 2)]['index'])
    print(f'Number of selected cols {len(sel_cols)} of {barycenters_data_df.shape[1]}')

    # select only interest columns (including poly_index, icc_code and tile_name)
    barycenters_data_df_sel = barycenters_data_df[sel_cols]

    # sort by column
    barycenters_data_df_sel = barycenters_data_df_sel.reindex(sorted(barycenters_data_df_sel.columns), axis=1)

    # fix nan values
    barycenters_data_df_sel_interp = barycenters_data_df_sel.drop(
        columns=['poly_index', 'tile_name', 'icc_code']).interpolate(option='nearest', axis=1)  # time
    barycenters_data_df_sel_all = barycenters_data_df_sel_interp.bfill(axis=1)
    barycenters_data_df_sel_all = barycenters_data_df_sel_all.ffill(axis=1)

    barycenters_data_df_sel_all = barycenters_data_df_sel_all.dropna(how='all')

    return barycenters_data_df_sel_all


def prepare_training(barycenters_data_df_sel_all, aoi, year, icc_code):
    df = barycenters_data_df_sel_all.sample(frac=1.0, random_state=RANDOM_SEED)
    print(f'training df shape {df.shape}')

    data_file = f's3://agrisen-cog-others/dtw_data/barycenters_lstmae/{aoi}/{year}/barycenter_poly_{icc_code}.parquet'

    df.to_parquet(data_file, storage_options={
        "key": key,
        "secret": secret,
        "client_kwargs": {"endpoint_url": endpoint_url}
    })
    train_df, val_df = train_test_split(
        df,
        test_size=0.3,
        random_state=RANDOM_SEED
    )

    print(f'split {train_df.shape}, {val_df.shape}')

    def create_dataset2(df):
        sequences = torch.tensor(df.astype(np.float32).to_numpy()).unsqueeze(2).float()
        print(sequences.shape)
        label = torch.full((sequences.shape[0],), int(icc_code))
        ds = TensorDataset(sequences, label)
        return ds

    train_ds = create_dataset2(train_df)
    val_ds = create_dataset2(val_df)

    def selecte_batch_size(num):
        if num > 1000:
            batch_size = 500
        elif num > 1000:
            batch_size = 50
        else:
            batch_size = 1
        return batch_size

    batch_size_train = selecte_batch_size(train_df.shape[0])
    print(f'train batch size {batch_size_train}')
    train_dl = DataLoader(train_ds, batch_size=batch_size_train,
                          shuffle=True)  # data (batch_size, num_ts) (1000, 44), label (batzh_size ), (1000)
    val_dl = DataLoader(val_ds, batch_size=batch_size_train, shuffle=True)

    all_ds = create_dataset2(df)
    all_dl = DataLoader(all_ds, batch_size=1,
                        shuffle=False)

    return train_dl, val_dl, all_ds


class Encoder(nn.Module):

    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(Encoder, self).__init__()

        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

        self.rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.rnn2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True
        )

    def forward(self, x):
        # x = x.reshape((1, self.seq_len, self.n_features))
        # print(x.shape)
        x, (_, _) = self.rnn1(x)
        # print(x.shape)
        x, (hidden_n, _) = self.rnn2(x)

        # print(x.shape)
        # print(hidden_n.reshape((-1, self.embedding_dim)).shape)
        return hidden_n.reshape((-1, self.embedding_dim))


def train_model(model, train_loader, val_loader, n_epochs, seq_len, batch_size=20, ):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.L1Loss(reduction='sum').to(device)
    history = dict(train=[], val=[])

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000.0

    for epoch in range(1, n_epochs + 1):
        model = model.train()

        train_losses = []
        # for seq_true in train_dataset:
        for seq_true, _ in train_loader:
            optimizer.zero_grad()

            seq_true = seq_true.to(device)
            seq_pred = model(seq_true)
            ##print('in train')
            # print(seq_true.shape)
            # print(seq_pred.shape)

            loss = criterion(seq_pred, seq_true)
            # loss = criterion(seq_pred.view(batch_size * seq_len, -1),seq_true.view(batch_size * seq_len))

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        val_losses = []
        model = model.eval()
        with torch.no_grad():
            # for seq_true in val_dataset:
            for seq_true, _ in val_loader:
                seq_true = seq_true.to(device)
                seq_pred = model(seq_true)

                loss = criterion(seq_pred, seq_true)
                # loss = criterion(seq_pred.view(batch_size*seq_len, -1), seq_true.view(batch_size * seq_len))

                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        history['train'].append(train_loss)
        history['val'].append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

        print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')

    model.load_state_dict(best_model_wts)
    return model.eval(), history


class Decoder(nn.Module):

    def __init__(self, seq_len, input_dim=64, n_features=1):
        super(Decoder, self).__init__()

        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features

        self.rnn1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=input_dim,
            num_layers=1,
            batch_first=True
        )

        self.rnn2 = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.output_layer = nn.Linear(self.hidden_dim, n_features)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = x.repeat(1, self.seq_len, self.n_features)

        x, (hidden_n, cell_n) = self.rnn1(x)
        x, (hidden_n, cell_n) = self.rnn2(x)
        x = x.reshape((-1, self.seq_len, self.hidden_dim))

        return self.output_layer(x)


class RecurrentAutoencoder(nn.Module):

    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(RecurrentAutoencoder, self).__init__()

        self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)
        self.decoder = Decoder(seq_len, embedding_dim, n_features).to(device)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x


def predict(model, dataset):
    predictions, losses = [], []
    criterion = nn.L1Loss(reduction='sum').to(device)
    with torch.no_grad():
        model = model.eval()
        for seq_true, _ in dataset:
            seq_true = seq_true.to(device)
            seq_pred = model(seq_true)
            loss = criterion(seq_pred, seq_true)
            predictions.append(seq_pred.cpu().numpy().flatten())
            losses.append(loss.item())
    return predictions, losses


def save_pred(output_file, pred_losses):
    pred_plot = sns.distplot(pred_losses, bins=50, kde=True)
    fig_pred = pred_plot.get_figure()
    fig_pred.savefig(output_file)


def run_anomaly_detection(aoi, year, input_dir, output_dir, icc_code_list):
    print(device)
    for icc_code in icc_code_list:
        barycenters_data_df_sel_all = prepare_file(input_dir, aoi, year, icc_code)
        train_dl, val_dl, all_dl = prepare_training(barycenters_data_df_sel_all, aoi, year, icc_code)
        seq_len = next(iter(train_dl))[0].shape[1]
        print(f'seq len {seq_len}')
        # seq_len = 24
        n_features = 1
        model = RecurrentAutoencoder(seq_len, n_features, 128)
        model = model.to(device)

        model, history = train_model(
            model,
            train_dl,
            val_dl,
            n_epochs=150,
            seq_len=seq_len,
            batch_size=1
        )

        output_path = f'{output_dir}/models/model_{icc_code}'
        if path.exists(output_path) == False:
            os.makedirs(output_path)
        MODEL_PATH = f'{output_path}/model_{icc_code}.pth'
        torch.save(model, MODEL_PATH)

        history_path = f'{output_path}/history_model_{icc_code}.json'

        with open(history_path, 'w') as f:
            json.dump(history, f)

        plt.clf()

        ax = plt.figure().gca()

        ax.plot(history['train'])
        ax.plot(history['val'])
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['train', 'val'])
        plt.title('Loss over training epochs')
        plt.savefig(f'{output_path}/history_plot_{icc_code}.png')

        # predict
        predictions, pred_losses = predict(model, all_dl)
        predictions_path = f'{output_path}/predictions_model_{icc_code}.npy'
        pred_losses_path = f'{output_path}/pred_losses_model_{icc_code}.json'

        with open(predictions_path, 'wb') as f:
            np.save(f, predictions)
        with open(pred_losses_path, 'w') as f:
            json.dump(pred_losses, f)

        plt.clf()

        pred_plot = sns.distplot(pred_losses, bins=50, kde=True)
        fig_pred = pred_plot.get_figure()
        fig_pred.savefig(f'{output_path}/pred_losses_plot_{icc_code}.png')


def main():
    # Parse user arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--aoi', type=str, default=None, required=False,
                        help='Name of the AOI to process.')
    parser.add_argument('--year', type=int, default=None, required=False,
                        help='Year of the AOI to process.')

    parser.add_argument('--input_dir', default=None, required=False,
                        help='S3 path to read the input tiles.')

    parser.add_argument('--output_dir', default=None, required=False,
                        help='Path to save the results.')

    parser.add_argument('--icc_code_list', default=None, required=False, nargs='+',
                        help='ICC Codes to run')

    args = parser.parse_args()

    run_anomaly_detection(args.aoi, args.year, args.input_dir, args.output_dir, args.icc_code_list)


if __name__ == '__main__':
    main()
