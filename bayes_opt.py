# bayes_opt.py
import os
import shutil
import yaml
import argparse
import optuna
from train_save_history import main, read_cfg_file, _get_log_dir
from config import args
import pandas as pd
import matplotlib.pyplot as plt

# MODE = "search"
MODE = "train"
# MODE = "search_train"

FINAL_CONFIG_FILES = [
    # "/path/to/your/other_config1.yaml",

]

def plot_bayes_search(csv_path, save_path=None, dpi=600):
    df = pd.read_csv(csv_path)
    df = df.sort_values('number')
    df['best_so_far'] = df['value'].cummin()

    best_idx   = df['value'].idxmin()
    best_trial = int(df.at[best_idx, 'number'])
    best_mae   = df.at[best_idx, 'value']

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times']
    plt.rcParams['pdf.fonttype'] = 42

    plt.figure(figsize=(8, 4))
    plt.plot(df['number'], df['value'],
             label='This Trial', color='#0072B2', zorder=1.5)
    plt.plot(df['number'], df['best_so_far'],
             label='Best So Far', color='#E69F00', zorder=1.5)
    plt.scatter([best_trial], [best_mae],
                s=50, color='#009E73', label='Best Point', zorder=2)

    plt.xlabel('Trial')
    plt.ylabel('Validation MAE')
    plt.title('Bayesian Optimization Progress')
    plt.legend()

    plt.annotate(
        f'Trial {best_trial}, MAE = {best_mae:.2f}',
        xy=(best_trial, best_mae),
        xytext=(-40, 8),
        textcoords='offset points',
        ha='right',
        va='top',
        color='#009E73',
        arrowprops=dict(
            arrowstyle='->',
            color='#009E73',
            lw=2,
            shrinkA=5,
            shrinkB=5
        ),
        bbox=dict(
            boxstyle='round,pad=0.3',
            edgecolor='#009E73',
            facecolor='white',
            linewidth=1
        ),
        zorder=3
    )

    if save_path:
        plt.savefig(save_path, format='pdf', dpi=dpi)
        print(f"Saved figure to {save_path} (dpi={dpi})")
    plt.show()

if __name__ == '__main__':
    base_cfg_path = args.config_filename
    base_cfg = read_cfg_file(base_cfg_path)

    bayes_dir = os.path.join(base_cfg['base_dir'], 'bayes_study')
    os.makedirs(bayes_dir, exist_ok=True)

    storage = f"sqlite:///{os.path.abspath(bayes_dir)}/study.db"

    if MODE in ("search", "search_train"):
        study = optuna.create_study(
            study_name='hyperparam_search_nothing_GCGRU',
            storage=storage,
            direction='minimize',
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.MedianPruner()
        )

        def objective(trial):
            cfg = read_cfg_file(base_cfg_path)
            cfg['base_dir'] = bayes_dir
            cfg.setdefault('train', {})['trial_number'] = trial.number

            cfg['train']['base_lr']      = trial.suggest_loguniform('base_lr',     1e-5, 1e-2)
            cfg['model']['dropout_prop'] = trial.suggest_uniform('dropout_prop',   0.0, 0.5)
            cfg['model']['rnn_units']    = trial.suggest_categorical('rnn_units', [64, 96, 128, 192])
            cfg['model']['channel']      = trial.suggest_categorical('channel',   [256, 512, 768])
            cfg['model']['PINN_value']   = trial.suggest_loguniform('PINN_value', 1e-3, 1e-1)

            cfg['model']['additional_distribution_dim'] = trial.suggest_categorical('additional_distribution_dim', [2, 3, 4, 5])
            cfg['model']['additional_section_feature_dim'] = trial.suggest_categorical('additional_section_feature_dim', [5, 6, 7, 8])

            cfg['train']['epochs'] = 40

            tmp_cfg = os.path.join(bayes_dir, f'temp_config_{trial.number}.yaml')
            with open(tmp_cfg, 'w') as f:
                yaml.dump(cfg, f)

            trial_args = argparse.Namespace(config_filename=tmp_cfg)
            val_mae = main(trial_args)
            os.remove(tmp_cfg)
            return val_mae

        study.optimize(objective, n_trials=30)

        df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
        csv_file = os.path.join(bayes_dir, 'trials.csv')
        df.to_csv(csv_file, index=False)
        png_file = os.path.join(bayes_dir, 'bayes_progress.pdf')
        print("bayes_dir =", bayes_dir)
        print("csv_file =", csv_file)
        print("png_file =", png_file)
        plot_bayes_search(csv_file, png_file)
        print('Best trial (val_mae):', study.best_trial.value)
        print('Best params:', study.best_trial.params)

    if MODE in ("train", "search_train"):
        orig_epochs = read_cfg_file(base_cfg_path)['train']['epochs']
        study = optuna.load_study(
            study_name='hyperparam_search_nothing_GCGRU',
            storage=storage
        )
        best_params = study.best_trial.params
        configs = FINAL_CONFIG_FILES or [base_cfg_path]

        for cfg_path in configs:
            cfg_train = read_cfg_file(cfg_path)
            replaced = {}
            for k, v in study.best_trial.params.items():
                if   k == 'base_lr':      cfg_train['train']['base_lr']   = v
                elif k == 'dropout_prop': cfg_train['model']['dropout_prop'] = v
                elif k == 'rnn_units':    cfg_train['model']['rnn_units']   = v
                elif k == 'channel':      cfg_train['model']['channel']     = v
                elif k == 'PINN_value':   cfg_train['model']['PINN_value']  = v
                elif k == 'additional_distribution_dim':
                    cfg_train['model']['additional_distribution_dim'] = v
                elif k == 'additional_section_feature_dim':
                    cfg_train['model']['additional_section_feature_dim'] = v

            cfg_train['train']['epochs'] = orig_epochs
            cfg_train['_replaced_params'] = replaced
            name = os.path.splitext(os.path.basename(cfg_path))[0]

            final_base = os.path.join(bayes_dir, 'final_train')
            os.makedirs(final_base, exist_ok=True)

            cfg_train['base_dir'] = final_base

            shutil.copy(cfg_path, os.path.join(final_base, os.path.basename(cfg_path)))

            final_cfg_path = os.path.join(final_base, os.path.basename(base_cfg_path))
            with open(final_cfg_path, 'w') as f:
                yaml.dump(cfg_train, f)

            print(f"Starting full training with config {cfg_path}, logs → {final_base}")
            args_final = argparse.Namespace(config_filename=final_cfg_path)
            final_mae = main(args_final)
            print(f'Final training done, best val MAE: {final_mae}')
