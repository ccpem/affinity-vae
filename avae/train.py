import logging
import os

import lightning as lt
import numpy as np
import pandas as pd
import torch

from . import utils_learning, utils_vis, vis
from .cyc_annealing import setup_annealing
from .data import load_data
from .loss import AVAELoss
from .models import model_setup
from .utils import as_list, latest_file
from .utils_gpu import setup_gpus
from .utils_learning import (
    build_meta_df,
    combine_meta_df,
    configure_optimiser,
    log_progress,
)


def train(params):
    """Function to train an AffinityVAE model. The inputs are training configuration parameters. In this function the
    data is loaded, selected and split into training, validation and test sets, the model is initialised and trained
    over epochs, the results are evaluated visualised and saved and the epoch level with a frequency configured with
    input parameters.

    Parameters
    ----------
    params : object
        A pydantic validated object containing all the training configuration parameters as attributes.
    """

    # start from plots/ in case an earlier run in this process failed mid-epoch
    utils_vis.set_vis_dir(None)
    utils_vis.set_run_id(params.date_time_run)
    # Lightning logs "Seed set" and "Initializing distributed" from every GPU
    # process; keep its info messages from the first one only
    if os.environ.get("LOCAL_RANK", "0") != "0":
        logging.getLogger("lightning.fabric").setLevel(logging.WARNING)
    lt.pytorch.seed_everything(42)

    # ############################### GPU SETUP ################################

    fabric = setup_gpus(
        gpu=params.gpu,
        gpu_devices=params.gpu_devices,
        strategy=params.strategy,
    )

    fabric.launch()
    device = fabric.device
    rank_zero = fabric.global_rank == 0

    # DDP intentionally stashes autograd nodes across streams; silence the
    # resulting benign AccumulateGrad stream-mismatch warning if available.
    if hasattr(
        torch.autograd.graph, "set_warn_on_accumulate_grad_stream_mismatch"
    ):
        torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)

    # ############################### ANNEALING ###############################

    beta_arr = setup_annealing(
        epochs=params.epochs,
        value_max=params.beta,
        value_min=params.beta_min,
        cyc_method=params.cyc_method_beta,
        n_cycle=params.beta_cycle,
        ratio=params.beta_ratio,
        cycle_load=params.beta_load,
    )

    gamma_arr = setup_annealing(
        epochs=params.epochs,
        value_max=params.gamma,
        value_min=params.gamma_min,
        cyc_method=params.cyc_method_gamma,
        n_cycle=params.gamma_cycle,
        ratio=params.gamma_ratio,
        cycle_load=params.gamma_load,
    )
    if rank_zero and params.vis_cyc:
        vis.plot_cyc_variable(
            beta_arr,
            "beta",
            vis_format=params.vis_format,
            vis_print=params.vis_print,
        )
        vis.plot_cyc_variable(
            gamma_arr,
            "gamma",
            vis_format=params.vis_format,
            vis_print=params.vis_print,
        )

    # ############################### DATA ###############################
    trains, vals, tests, affinity_matrix, data_dim = load_data(
        eval=False,
        datapath=params.datapath,
        datatype=params.datatype,
        lim=params.limit,
        splt=params.split,
        batch=params.batch,
        no_val_drop=params.no_val_drop,
        affinity_path=params.affinity,
        classes=params.classes,
        gaussian_blur=params.gaussian_blur,
        normalise=params.normalise,
        shift_min=params.shift_min,
        rescale=params.rescale,
        vis_his=params.vis_his,
        vis_aff=params.vis_aff,
        vis_format=params.vis_format,
        vis_print=params.vis_print,
        fabric=fabric,
    )

    if len(trains) == 0:
        error = (
            "Training size is 0. Check the 'batch_size' parameter if it's not larger than your existing data."
            " If you're running distributed learning, each GPU gets an equal fraction of the data, adjust your"
            " batch size accordingly. If you are using a small dataset, consider using a smaller number of GPUs"
            " or running on a single GPU. Finally check 'limit' parameter if it is not set to smaller than batch_size."
        )
        logging.error(error)
        raise RuntimeError(error)

    dshape = next(iter(trains))[0].shape[2:]
    pose = not (params.pose_dims == 0)

    # ############################### MODEL ###############################
    vae = model_setup(
        model_type=params.model,
        input_shape=dshape,
        channels=params.channels,
        depth=params.depth,
        latent_dims=params.latent_dims,
        pose_dims=params.pose_dims,
        bnorm_encoder=params.bnorm_encoder,
        bnorm_decoder=params.bnorm_decoder,
        n_splats=params.n_splats,
        gsd_conv_layers=params.gsd_conv_layers,
        device=device,
        filters=params.filters,
        verbose=rank_zero,
    )

    # ################################# LOSS #################################

    loss = AVAELoss(
        device=device,
        beta=beta_arr,
        gamma=gamma_arr,
        lookup_aff=affinity_matrix,
        recon_loss=params.recon_loss,
        klreduction=params.klreduction,
    )

    # ############################### OPTIMISER ###############################
    optimizer = configure_optimiser(
        opt_method=params.opt_method, model=vae, learning_rate=params.learning
    )

    vae, optimizer = fabric.setup(vae, optimizer)

    # ############################### RESTARTS ################################
    e_start = 0
    t_history = []
    v_history = []
    best_val_loss = float("inf")
    best_epoch = None  # 1-based epoch of best_val_loss
    best_val_acc = None  # best validation accuracy so far
    best_acc_epoch = None  # 1-based epoch of best_val_acc
    if params.restart:
        if state is None:
            if not os.path.exists("states"):
                raise RuntimeError(
                    "There are no existing model states saved or provided either via the state flag or in the config. Unable to evaluate."
                )
            else:
                state = latest_file("states", ".pt")
                state = os.path.join("states", state)

        checkpoint = torch.load(state, weights_only=False)
        vae.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        e_start = checkpoint["epoch"]
        t_history = checkpoint["t_loss_history"]
        v_history = checkpoint["v_loss_history"]
        if params.vis_best and checkpoint.get("best_val_loss") is not None:
            best_val_loss = float(checkpoint["best_val_loss"])
            best_epoch = checkpoint.get("best_epoch")
        if params.vis_best and checkpoint.get("best_val_acc") is not None:
            best_val_acc = float(checkpoint["best_val_acc"])
            best_acc_epoch = checkpoint.get("best_acc_epoch")
        tracked = v_history[params.best_start - 1 :]
        losses = [float(v[0]) for v in tracked]
        if (
            params.vis_best
            and checkpoint.get("best_val_loss") is None
            and not np.all(np.isnan(losses))
        ):
            best_val_loss = float(np.nanmin(losses))
            best_epoch = params.best_start + int(np.nanargmin(losses))

    # ########################## TRAINING LOOP ################################

    if rank_zero:
        logging.info("\n")
        logging.info(
            "############################################### TRAINING"
        )

    for epoch in range(e_start, params.epochs):

        # populate loss with new epoch
        t_history.append(np.zeros(4))
        v_history.append(np.zeros(4))

        # Per-rank epoch metadata buffers grouped by mode.
        filename_train, filename_val, filename_test = [], [], []
        meta_train, meta_val, meta_test = [], [], []

        # create holders for data, labels, and latent spaces
        x_train, x_val, x_test = [], [], []
        xhat_train, xhat_val, xhat_test = [], [], []

        y_train, y_val, y_test = [], [], []
        z_train, z_val, z_test = [], [], []
        c_train, c_val, c_test = [], [], []

        if pose:
            p_train, p_val, p_test = [], [], []

        # ########################## TRAINING #################################
        vae.train()
        for batch_number, (x, ys, aff, meta_data) in enumerate(trains):

            # get data in the right device
            x, aff = x.to(device), aff.to(device)
            x = x.to(torch.float32)

            # forward
            x_hat, lat_mu, lat_logvar, lat, lat_pose = vae(x)
            history_loss = loss(
                x, x_hat, lat_mu, lat_logvar, epoch, batch_aff=aff
            )

            # record loss
            for i in range(len(t_history[-1])):
                t_history[-1][i] += history_loss[i].item()
            if rank_zero:
                log_progress(
                    "Epoch: [%d/%d] | Batch: [%d/%d]"
                    % (epoch + 1, params.epochs, batch_number + 1, len(trains))
                )

            # backwards
            fabric.backward(history_loss[0])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            z_train.extend(lat_mu.cpu().detach().numpy())  # store latents
            y_train.extend(as_list(ys))
            c_train.extend(lat_logvar.cpu().detach().numpy())
            if pose:
                p_train.extend(lat_pose.cpu().detach().numpy())

            filename_train.extend(as_list(meta_data.get("filename", [])))
            meta_train.extend(as_list(meta_data.get("meta", [])))
            x_train.extend(as_list(meta_data.get("image", [])))
            xhat_train.extend(vis.format(x_hat, data_dim))

        # mean over the batches of all ranks, logged once
        t_history[-1] = utils_learning.mean_over_ranks(
            t_history[-1], len(trains), fabric
        )
        if rank_zero:
            logging.info(
                "Training : Epoch: [%d/%d] | Loss: %f | Recon: %f | "
                "KLdiv: %f | Affin: %f | Beta: %f | Gamma: %f"
                % (
                    epoch + 1,
                    params.epochs,
                    *t_history[-1],
                    beta_arr[epoch],
                    gamma_arr[epoch],
                )
            )
        # ########################## VAL ######################################
        vae.eval()
        with torch.inference_mode():
            for batch_number, (v, ys, aff, meta_data) in enumerate(vals):

                # get data in the right device
                v, aff = v.to(device), aff.to(device)
                v = v.to(torch.float32)

                # forward
                v_hat, v_mu, v_logvar, vlat, vlat_pos = vae(v)
                v_history_loss = loss(
                    v, v_hat, v_mu, v_logvar, epoch, batch_aff=aff
                )

                # record loss
                for i in range(len(t_history[-1])):
                    v_history[-1][i] += v_history_loss[i].item()
                if rank_zero:
                    log_progress(
                        "Epoch: [%d/%d] | Batch: [%d/%d]"
                        % (
                            epoch + 1,
                            params.epochs,
                            batch_number + 1,
                            len(vals),
                        )
                    )

                z_val.extend(v_mu.cpu().detach().numpy())  # store latents
                y_val.extend(as_list(ys))
                c_val.extend(v_logvar.cpu().detach().numpy())
                if pose:
                    p_val.extend(vlat_pos.cpu().detach().numpy())

                filename_val.extend(as_list(meta_data.get("filename", [])))
                meta_val.extend(as_list(meta_data.get("meta", [])))
                x_val.extend(as_list(meta_data.get("image", [])))
                xhat_val.extend(vis.format(v_hat, data_dim))
            # mean over the batches of all ranks, so every rank logs, plots
            # and tracks the best epoch with the same value
            v_history[-1] = utils_learning.mean_over_ranks(
                v_history[-1], len(vals), fabric
            )
            global_val_loss = float(v_history[-1][0])

            # best before this epoch, shown at the end of the validation log
            last_best_parts = []
            if best_acc_epoch is not None and best_val_acc is not None:
                last_best_parts.append(
                    "%d %.0f%% acc" % (best_acc_epoch, 100 * best_val_acc)
                )
            if best_epoch is not None:
                last_best_parts.append("%d loss" % best_epoch)
            last_best = (
                " | Best: " + ", ".join(last_best_parts)
                if params.vis_best and last_best_parts
                else ""
            )

            # lowest validation total loss so far (only tracked with vis_best),
            # computed over all ranks so all take the same plotting branches
            is_best_loss = bool(
                params.vis_best
                and epoch + 1 >= params.best_start
                and global_val_loss < best_val_loss
            )
            if is_best_loss:
                best_val_loss = global_val_loss
                best_epoch = epoch + 1

            if rank_zero:
                logging.info(
                    "Validation : Epoch: [%d/%d] | Loss: %f | Recon: %f | "
                    "KLdiv: %f | Affin: %f | Beta: %f | Gamma: %f%s"
                    % (
                        epoch + 1,
                        params.epochs,
                        *v_history[-1],
                        beta_arr[epoch],
                        gamma_arr[epoch],
                        last_best,
                    )
                )

            # ########################## TEST #####################################
            if params.freq_eval != 0 and (epoch + 1) % params.freq_eval == 0:
                for batch_number, (t, ys, aff, meta_data) in enumerate(
                    tests
                ):  # tests empty if no 'test' dir
                    # get data in the right device
                    t, aff = t.to(device), aff.to(device)
                    t = t.to(torch.float32)

                    # forward
                    t_hat, t_mu, t_logvar, tlat, tlat_pose = vae(t)

                    z_test.extend(t_mu.cpu().detach().numpy())  # store latents
                    y_test.extend(as_list(ys))
                    c_test.extend(t_logvar.cpu().detach().numpy())
                    if pose:
                        p_test.extend(tlat_pose.cpu().detach().numpy())

                    filename_test.extend(
                        as_list(meta_data.get("filename", []))
                    )
                    meta_test.extend(as_list(meta_data.get("meta", [])))
                    x_test.extend(as_list(meta_data.get("image", [])))
                    xhat_test.extend(vis.format(t_hat, data_dim))

                    if rank_zero:
                        log_progress(
                            "Epoch: [%d/%d] | Batch: [%d/%d]"
                            % (
                                epoch + 1,
                                params.epochs,
                                batch_number + 1,
                                len(tests),
                            )
                        )
                if rank_zero:
                    logging.info(
                        "Evaluation : Epoch: [%d/%d]"
                        % (epoch + 1, params.epochs)
                    )
            if rank_zero:
                logging.info("\n")  # end of training round

        # ########################## COMPUTE RESULTS ##########################
        track_acc = params.vis_best and epoch + 1 >= params.best_start
        # accuracy on the latents of all ranks (collective gather): every
        # epoch from best_start for best tracking, and every freq_acc epochs
        # for vis_acc; the accuracy plots below reuse the result
        acc_cache = None
        is_best_acc = False
        if track_acc or (
            params.vis_acc
            and params.freq_acc != 0
            and (epoch + 1) % params.freq_acc == 0
        ):
            acc_gathered = utils_learning.combine_accuracy_data(
                z_train,
                y_train,
                z_val,
                y_val,
                rank_zero=rank_zero,
                world_size=fabric.world_size,
            )
            acc_val = 0.0
            if rank_zero:
                assert acc_gathered is not None
                (
                    acc_train,
                    acc_val,
                    _,
                    acc_ypred_train,
                    acc_ypred_val,
                ) = utils_learning.accuracy(
                    *acc_gathered,
                    classifier=params.classifier,
                )
                acc_cache = (acc_gathered, acc_ypred_train, acc_ypred_val)
                logging.info(
                    "\n------------------->>> Overall accuracy: "
                    "Train: %f | Val: %f\n" % (acc_train, acc_val)
                )
            acc_val = float(
                fabric.broadcast(
                    torch.tensor(float(acc_val), device=fabric.device),
                    src=0,
                )
            )
            if track_acc:
                is_best_acc = best_val_acc is None or acc_val > best_val_acc
                if is_best_acc:
                    best_val_acc = acc_val
                    best_acc_epoch = epoch + 1
        is_best = is_best_loss or is_best_acc

        emb_due = params.vis_emb and (
            (params.freq_emb != 0 and (epoch + 1) % params.freq_emb == 0)
            or is_best
        )
        dynamic_due = params.vis_dynamic and (
            (
                params.freq_dynamic != 0
                and (epoch + 1) % params.freq_dynamic == 0
            )
            or is_best
        )
        # metadata of all ranks, for the t-SNE, the embeddings and the saved
        # states
        needs_meta_df = (
            emb_due
            or dynamic_due
            or (params.freq_sta != 0 and (epoch + 1) % params.freq_sta == 0)
        )
        if needs_meta_df:
            meta_df = build_meta_df(
                pose=pose,
                train={
                    "filename": filename_train,
                    "meta": meta_train,
                    "x": x_train,
                    "xhat": xhat_train,
                    "y": y_train,
                    "z": z_train,
                    "logvar": c_train,
                    "pose": p_train if pose else None,
                },
                val={
                    "filename": filename_val,
                    "meta": meta_val,
                    "x": x_val,
                    "xhat": xhat_val,
                    "y": y_val,
                    "z": z_val,
                    "logvar": c_val,
                    "pose": p_val if pose else None,
                },
                test={
                    "filename": filename_test,
                    "meta": meta_test,
                    "x": x_test,
                    "xhat": xhat_test,
                    "y": y_test,
                    "z": z_test,
                    "logvar": c_test,
                    "pose": p_test if pose else None,
                },
            )

            # Dynamic plots and save should use metadata from all ranks.
            combined_meta_df = combine_meta_df(
                meta_df=meta_df,
                rank_zero=rank_zero,
                world_size=fabric.world_size,
            )

        # t-SNE of the latents (and poses) for the embedding plots
        if rank_zero and (emb_due or dynamic_due):
            latent_columns = [
                col for col in combined_meta_df if col.startswith("lat")
            ]
            embedding_xs = combined_meta_df[latent_columns].to_numpy()
            embedding_ys = combined_meta_df["id"].to_numpy().copy()
            embedding_ys = embedding_ys.astype(object)
            test_mask = combined_meta_df["mode"].to_numpy() == "tst"
            embedding_ys[test_mask] = utils_vis.plot_labels(
                embedding_ys[test_mask], "test"
            )
            latent_embedding = utils_learning.tsne_embedding(embedding_xs)
            if emb_due and pose:
                pose_columns = [
                    col for col in combined_meta_df if col.startswith("pos")
                ]
                pose_xs = combined_meta_df[pose_columns].to_numpy()
                pose_embedding = utils_learning.tsne_embedding(pose_xs)
            logging.info("")

        # ########################## VISUALISE ################################
        # what is plotted this epoch (the same on every rank)
        acc_due = params.vis_acc and (
            (params.freq_acc != 0 and (epoch + 1) % params.freq_acc == 0)
            or is_best
        )
        los_due = params.vis_los and epoch > 0
        rec_due = params.vis_rec and (
            (params.freq_rec != 0 and (epoch + 1) % params.freq_rec == 0)
            or is_best
        )
        dis_due = params.vis_dis and (
            (params.freq_dis != 0 and (epoch + 1) % params.freq_dis == 0)
            or is_best
        )
        pos_due = (
            pose
            and params.vis_pos
            and (
                (params.freq_pos != 0 and (epoch + 1) % params.freq_pos == 0)
                or is_best
            )
        )
        int_due = params.vis_int and (
            (params.freq_int != 0 and (epoch + 1) % params.freq_int == 0)
            or is_best
        )
        sim_due = params.vis_sim and (
            (params.freq_sim != 0 and (epoch + 1) % params.freq_sim == 0)
            or is_best
        )
        vis_due = any(
            (
                acc_due,
                los_due,
                rec_due,
                emb_due,
                dynamic_due,
                dis_due,
                pos_due,
                int_due,
                sim_due,
            )
        )

        # plot folders, only set up when something is plotted this epoch
        if vis_due and is_best:
            # save into the best folder (keeping only the latest of each
            # kind), copying each file to the other best folder, the epoch
            # folder and plots/ as it is saved
            kinds = [
                k
                for k, new in (("loss", is_best_loss), ("acc", is_best_acc))
                if new
            ]
            if rank_zero:
                for kind in kinds:
                    utils_vis.clear_best_dirs(kind)
            copy_to = [utils_vis.vis_dir(epoch + 1, best=k) for k in kinds[1:]]
            if params.vis_epochs:
                copy_to.append(utils_vis.vis_dir(epoch + 1))
            copy_to.append(utils_vis.vis_dir(None))
            utils_vis.set_vis_dir(
                epoch + 1, best=kinds[0], copy_to=tuple(copy_to)
            )
        elif vis_due and params.vis_epochs:
            # save into the epoch folder, copying each file to plots/
            utils_vis.set_vis_dir(
                epoch + 1, copy_to=(utils_vis.vis_dir(None),)
            )
        else:
            utils_vis.set_vis_dir(None)
        if rank_zero:
            logging.info("")

        # visualise loss
        if rank_zero and los_due:
            p = [
                len(trains),
                params.depth,
                params.channels,
                params.latent_dims,
                params.learning,
                beta_arr[epoch],
                gamma_arr[epoch],
            ]
            vis.loss_plot(
                epoch + 1,
                beta_arr[: epoch + 1],
                gamma_arr[: epoch + 1],
                t_history,
                v_history,
                p=p,
                vis_format=params.vis_format,
                vis_print=params.vis_print,
            )

        # visualise accuracy: confusion and F1 scores (computed under
        # COMPUTE RESULTS; is_best implies it was computed this epoch)
        if rank_zero and acc_due and acc_cache is not None:
            (_, y_train_all, _, y_val_all), ypred_train, ypred_val = acc_cache
            vis.confusion_plot(
                y_train_all,
                ypred_train,
                y_val_all,
                ypred_val,
                epoch=epoch,
                vis_format=params.vis_format,
                vis_print=params.vis_print,
            )

            vis.f1_plot(
                y_train_all,
                ypred_train,
                y_val_all,
                ypred_val,
                epoch=epoch,
                vis_format=params.vis_format,
                vis_print=params.vis_print,
            )

        # visualise reconstructions - last batch
        if rank_zero and rec_due:
            vis.recon_plot(
                x,
                x_hat,
                y_train,
                data_dim,
                mode="trn",
                vis_format=params.vis_format,
                vis_print=params.vis_print,
            )
            vis.recon_plot(
                v,
                v_hat,
                y_val,
                data_dim,
                mode="val",
                vis_format=params.vis_format,
                vis_print=params.vis_print,
            )

        # similarity, traversals and interpolations use the latents of all
        # ranks, so they see the whole training set and not one GPU's share
        if dis_due or pos_due or int_due or sim_due:
            gathered = utils_learning.combine_rank_data(
                (
                    z_train,
                    y_train,
                    p_train if pose else [],
                    z_val,
                    y_val,
                    p_val if pose else [],
                ),
                rank_zero=rank_zero,
                world_size=fabric.world_size,
            )
            if rank_zero:
                assert gathered is not None
                z_trn, y_trn, p_trn, z_vld, y_vld, p_vld = gathered
                if not pose:
                    p_trn = p_vld = None

        # a vis_class that is not in the data stops every rank, not only
        # rank 0 (which makes the per-class plots), so none is left waiting
        if params.vis_class and (dis_due or pos_due):
            class_error = ""
            if rank_zero:
                try:
                    utils_vis.requested_classes(
                        y_trn, params.vis_class, "vis_class"
                    )
                except ValueError as e:
                    class_error = str(e)
            class_error = fabric.broadcast(class_error, src=0)
            if class_error:
                raise ValueError(class_error)

        # visualise mean and logvar similarity matrix
        if rank_zero and sim_due:
            vis.latent_space_similarity_plot(
                z_trn,
                np.array(y_trn),
                mode="trn",
                epoch=epoch,
                affinity_matrix=params.affinity,
                vis_format=params.vis_format,
                vis_print=params.vis_print,
            )
            vis.latent_space_similarity_plot(
                z_vld,
                np.array(y_vld),
                mode="val",
                epoch=epoch,
                affinity_matrix=params.affinity,
                vis_format=params.vis_format,
                vis_print=params.vis_print,
            )

        # visualise embeddings
        if rank_zero and emb_due:
            vis.latent_embed_plot_tsne(
                embedding_xs,
                embedding_ys,
                epoch=epoch,
                vis_format=params.vis_format,
                vis_print=params.vis_print,
                embedding=latent_embedding,
            )

            if pose:
                vis.latent_embed_plot_tsne(
                    pose_xs,
                    embedding_ys,
                    epoch=epoch,
                    mode="pose",
                    vis_format=params.vis_format,
                    vis_print=params.vis_print,
                    embedding=pose_embedding,
                )

        if rank_zero and dynamic_due:
            # merge img and rec into one image for display in altair
            dynamic_meta_df = combined_meta_df.copy()
            dynamic_meta_df["image"] = dynamic_meta_df["image"].apply(
                vis.merge
            )
            vis.dyn_latentembed_plot(
                dynamic_meta_df, epoch, embedding=latent_embedding
            )

        # visualise latent disentanglement
        if rank_zero and dis_due:
            vis.latent_disentamglement_plot(
                dshape,
                z_trn,
                vae,
                device,
                poses=p_trn,
                vis_format=params.vis_format,
                vis_print=params.vis_print,
            )
            if params.vis_class:
                vis.latent_class_disentanglement_plot(
                    dshape,
                    z_trn,
                    y_trn,
                    params.vis_class,
                    vae,
                    device,
                    poses=p_trn,
                    vis_format=params.vis_format,
                    vis_print=params.vis_print,
                )

        # visualise pose disentanglement
        if rank_zero and pos_due:
            vis.pose_disentanglement_plot(
                dshape,
                z_trn,
                p_trn,
                vae,
                device,
                vis_format=params.vis_format,
                vis_print=params.vis_print,
            )

            if params.vis_class:
                vis.pose_class_disentanglement_plot(
                    dshape,
                    z_trn,
                    y_trn,
                    params.vis_class,
                    p_trn,
                    vae,
                    device,
                    vis_format=params.vis_format,
                    vis_print=params.vis_print,
                )

        # visualise interpolations
        if rank_zero and int_due:
            # train and val only, test latents are not used here
            xs = np.r_[z_trn, z_vld]
            ys = np.r_[y_trn, y_vld]
            ps = np.r_[p_trn, p_vld] if pose else None

            vis.interpolations_plot(
                dshape,
                xs,
                ys,
                vae,
                device,
                poses=ps,  # do we need val and test here?
                vis_format=params.vis_format,
                vis_print=params.vis_print,
            )
        # ########################## SAVE STATE ###############################
        if params.freq_sta != 0 and (epoch + 1) % params.freq_sta == 0:
            if rank_zero:
                if not os.path.exists("states"):
                    os.mkdir("states")

                mname = (
                    "avae_"
                    + str(params.date_time_run)
                    + "_E"
                    + str(epoch)
                    + "_"
                    + str(params.latent_dims)
                    + "_"
                    + str(params.pose_dims)
                    + ".pt"
                )

                logging.info(
                    "################################################################"
                )

                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": vae._original_module.state_dict(),
                        "model_class_object": vae._original_module,
                        "optimizer_state_dict": optimizer._optimizer.state_dict(),
                        "t_loss_history": t_history,
                        "v_loss_history": v_history,
                        "best_val_loss": (
                            best_val_loss if best_epoch is not None else None
                        ),
                        "best_epoch": best_epoch,
                        "best_val_acc": best_val_acc,
                        "best_acc_epoch": best_acc_epoch,
                    },
                    os.path.join("states", mname),
                )
                logging.info(
                    f"Saved model state: {mname} for restarting and evaluation "
                )

                filename = (
                    "meta_"
                    + str(params.date_time_run)
                    + "_E"
                    + str(epoch)
                    + "_L"
                    + str(params.latent_dims)
                    + "_P"
                    + str(params.pose_dims)
                    + ".pkl"
                )
                combined_meta_df.to_pickle(os.path.join("states", filename))

                logging.info(f"Saved meta file : {filename} for evaluation \n")
        fabric.barrier()

    utils_vis.set_vis_dir(None)
    utils_vis.set_run_id("")
