import torch
from torch import nn
import numpy as np
from copy import deepcopy
from tqdm import tqdm
class Trainer():
    def __init__(self, args, optimizer, lr_scheduler, loss_fn, evaluator, result_tracker, summary_writer, device, label_mean=None, label_std=None, ddp=False, local_rank=0, val_dataloader = None):
        self.args = args
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn
        self.evaluator = evaluator
        self.result_tracker = result_tracker
        self.summary_writer = summary_writer
        self.device = device
        self.label_mean = label_mean
        self.label_std = label_std
        self.ddp = ddp
        self.local_rank = local_rank
        if self.args.log_wandb:
            # Initialize wandb for experiment tracking and log hyperparameters from args
            import wandb
            self.wandb = wandb
            # get time, Mon_xx_hh_mm
            self.wandb.login(key='4709c292ef146d8b1d71fb9522a9e94b06cc6567')
            import time
            current_time =  time.strftime("%m_%d_%H_%M", time.localtime())
            run_name = f"{args.dataset}_lr{args.lr}_bs{args.batch_size}_{current_time}"
            self.wandb.init(project="cmx_kpgt", name=run_name, config=vars(args))



        self.best_loss = 10

    def _forward_epoch(self, model, batched_data, perturb=None):
        (smiles, g, ecfp, md, labels) = batched_data
        ecfp = ecfp.to(self.device)
        md = md.to(self.device)
        g = g.to(self.device)
        labels = labels.to(self.device)
        if perturb is None:
            predictions = model.forward_tune(g, ecfp, md)
        else:
            predictions = model.forward_tune(g, ecfp, md, perturb)
        return predictions, labels

    def train_epoch(self, model, train_loader, epoch_idx, val_loader):
        model.train()
        iter_loader = tqdm(train_loader, desc=f'Train Epoch {epoch_idx}', disable=(self.local_rank!=0))
        for batch_idx, batched_data in enumerate(iter_loader):


            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()
            predictions, labels = self._forward_epoch(model, batched_data)
            is_labeled = (~torch.isnan(labels)).to(torch.float32)
            labels = torch.nan_to_num(labels)
            if (self.label_mean is not None) and (self.label_std is not None):
                labels = (labels - self.label_mean)/self.label_std
            loss = (self.loss_fn(predictions, labels) * is_labeled).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            self.optimizer.step()

            # Default summary_writer
            if self.summary_writer is not None:
                self.summary_writer.add_scalar('Loss/train', loss, (epoch_idx-1)*len(train_loader)+batch_idx+1)
            
            if self.args.log_wandb:
                self.wandb.log({'Loss/train': loss}, step=(epoch_idx-1)*len(train_loader)+batch_idx+1)


    def fit(self, model, train_loader, val_loader, test_loader):
        best_val_result,best_test_result,best_train_result = self.result_tracker.init(),self.result_tracker.init(),self.result_tracker.init()
        best_epoch = 0
        for epoch in range(1, self.args.n_epochs+1):
            if self.ddp:
                train_loader.sampler.set_epoch(epoch)
            
            if self.local_rank == 0:
                val_result = self.eval(model, val_loader)
                test_result = self.eval(model, test_loader)
                train_result = self.eval(model, train_loader)
                if self.args.metric == 'cls-all':
                    print(f"Epoch {epoch}: Train ROC-AUC {train_result['rocauc']}, Val ROC-AUC {val_result['rocauc']}, Test ROC-AUC {test_result['rocauc']}")
                    if self.args.log_wandb:
                        self.wandb.log({'ROC-AUC/Train': train_result['rocauc'], 'ROC-AUC/Val': val_result['rocauc'], 'ROC-AUC/Test': test_result['rocauc']}, step=epoch)
                        self.wandb.log({'AP/Train': train_result['ap'], 'AP/Val': val_result['ap'], 'AP/Test': test_result['ap']}, step=epoch)
                        self.wandb.log({'ACC/Train': train_result['acc'], 'ACC/Val': val_result['acc'], 'ACC/Test': test_result['acc']}, step=epoch)
                    val_result, test_result, train_result = val_result['rocauc'], test_result['rocauc'], train_result['rocauc']

                if self.result_tracker.update(np.mean(best_val_result), np.mean(val_result)):
                    best_val_result = val_result
                    best_test_result = test_result
                    best_train_result = train_result
                    best_epoch = epoch
                    # save
                    import os
                    saves = os.path.join(self.args.save_path, f'{self.args.dataset}_best_auc{best_val_result}.pth')
                    torch.save(model.state_dict(), saves)
                print(np.mean(best_train_result), np.mean(best_val_result), np.mean(best_test_result))
                # DEBUG 
                # if epoch - best_epoch >= 20:
                #     break
                self.train_epoch(model, train_loader, epoch, val_loader)
            
        return np.mean(best_train_result), np.mean(best_val_result), np.mean(best_test_result)
    def eval(self, model, dataloader,direct=False):
    
        model.eval()
        predictions_all = []
        labels_all = []
        eval_loader = tqdm(dataloader, desc='Eval', disable=(self.local_rank!=0))
        for batch_idx, batched_data in enumerate(eval_loader):
            # DEBUG
            # if True:
            #     if batch_idx > 2:
            #         break
            predictions, labels = self._forward_epoch(model, batched_data)
            predictions_all.append(predictions.detach().cpu())
            labels_all.append(labels.detach().cpu())
        result = self.evaluator.eval(torch.cat(labels_all), torch.cat(predictions_all), direct=direct)
        return result


class FLAG_Trainer(Trainer):
    def __init__(self, args, d_node_hidden, optimizer, lr_scheduler, loss_fn, evaluator, result_tracker, summary_writer, device, label_mean=None, label_std=None, ddp=False, local_rank=0):
        super().__init__(args, optimizer, lr_scheduler, loss_fn, evaluator, result_tracker, summary_writer, device, label_mean=label_mean, label_std=label_std, ddp=ddp, local_rank=local_rank)
        self.d_node_hidden = d_node_hidden
    def train_epoch(self, model, train_loader, epoch_idx):
        model.train()
        iter_loader = tqdm(train_loader, desc=f'FLAG Train Epoch {epoch_idx}', disable=(self.local_rank!=0))
        for batch_idx, batched_data in enumerate(iter_loader):
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()
            perturb_shape = (batched_data[1].number_of_nodes(), self.d_node_hidden)
            perturb = torch.FloatTensor(
                *perturb_shape).uniform_(-self.args.flag_step_size, self.args.flag_step_size).to(self.device)
            perturb.requires_grad_()
            predictions, labels = self._forward_epoch(model, batched_data, perturb)
            
            is_labeled = (~torch.isnan(labels)).to(torch.float32)
            labels = torch.nan_to_num(labels)
            if (self.label_mean is not None) and (self.label_std is not None):
                labels = (labels - self.label_mean)/self.label_std
            loss = (self.loss_fn(predictions, labels) * is_labeled).mean()
            loss /= self.args.flag_m
            for _ in range(self.args.flag_m - 1):
                loss.backward()
                perturb_data = perturb.detach() + self.args.flag_step_size * torch.sign(perturb.grad.detach())
                perturb.data = perturb_data.data
                perturb.grad[:] = 0
                predictions, labels = self._forward_epoch(model, batched_data, perturb)
                is_labeled = (~torch.isnan(labels)).to(torch.float32)
                labels = torch.nan_to_num(labels)
                if (self.label_mean is not None) and (self.label_std is not None):
                    labels = (labels - self.label_mean)/self.label_std
                loss = (self.loss_fn(predictions, labels) * is_labeled).mean()
                loss /= self.args.flag_m
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            self.optimizer.step()
            if self.summary_writer is not None:
                self.summary_writer.add_scalar('Loss/train', loss, (epoch_idx-1)*len(train_loader)+batch_idx+1)


class SPRegularization(nn.Module):
    def __init__(self, source_model: nn.Module, target_model: nn.Module):
        super(SPRegularization, self).__init__()
        self.target_model = target_model
        self.source_weight = {}
        for name, param in source_model.named_parameters():
            if 'predictor' not in name:
                self.source_weight[name] = param.detach()
            else:
                pass

    def forward(self):
        output = 0.0
        for name, param in self.target_model.named_parameters():
            if name in self.source_weight.keys():
                output += 0.5 * torch.norm(param - self.source_weight[name]) ** 2
        return output

class L2SP_Trainer(Trainer):
    def __init__(self, args, optimizer, lr_scheduler, loss_fn, evaluator, result_tracker, summary_writer, device, label_mean=None, label_std=None, ddp=False, local_rank=0):
        super().__init__(args, optimizer, lr_scheduler, loss_fn, evaluator, result_tracker, summary_writer, device, label_mean=label_mean, label_std=label_std, ddp=ddp, local_rank=local_rank)
        self.args = args
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn
        self.evaluator = evaluator
        self.result_tracker = result_tracker
        self.summary_writer = summary_writer
        self.device = device
        self.label_mean = label_mean
        self.label_std = label_std
        self.ddp = ddp
        self.local_rank = local_rank

    def train_epoch(self, spr, model, train_loader, epoch_idx):
        model.train()
        iter_loader = tqdm(train_loader, desc=f'L2SP Train Epoch {epoch_idx}', disable=(self.local_rank!=0))
        for batch_idx, batched_data in enumerate(iter_loader):
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()
            predictions, labels = self._forward_epoch(model, batched_data)
            is_labeled = (~torch.isnan(labels)).to(torch.float32)
            labels = torch.nan_to_num(labels)
            if (self.label_mean is not None) and (self.label_std is not None):
                labels = (labels - self.label_mean)/self.label_std
            loss = (self.loss_fn(predictions, labels) * is_labeled).mean() + self.args.l2sp_weight*spr()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            self.optimizer.step()
            if self.summary_writer is not None:
                self.summary_writer.add_scalar('Loss/train', loss, (epoch_idx-1)*len(train_loader)+batch_idx+1)
        

    def fit(self, model, train_loader, val_loader, test_loader):
        source_model = deepcopy(model).to(self.device)
        spr = SPRegularization(source_model, model)
        best_val_result,best_test_result,best_train_result = self.result_tracker.init(),self.result_tracker.init(),self.result_tracker.init()
        best_epoch = 0
        for epoch in range(1, self.args.n_epochs+1):
            if self.ddp:
                train_loader.sampler.set_epoch(epoch)
            self.train_epoch(spr, model, train_loader, epoch)
            if self.local_rank == 0:
                val_result = self.eval(model, val_loader)
                test_result = self.eval(model, test_loader)
                train_result = self.eval(model, train_loader)
                if self.result_tracker.update(np.mean(best_val_result), np.mean(val_result)):
                    best_val_result = val_result
                    best_test_result = test_result
                    best_train_result = train_result
                    best_epoch = epoch
                if epoch - best_epoch >= 20:
                    break
        return np.mean(best_train_result), np.mean(best_val_result), np.mean(best_test_result)
    