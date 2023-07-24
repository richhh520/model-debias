#! /usr/bin/env python3

import torch
import time
import datetime
import numpy as np
import copy
import logging
import shutil
from pathlib import Path
from .influence_function import s_test, grad_z
from .utils import save_json, display_progress
import os




def calc_s_test(model, test_loader, train_loader, save=False, gpu=-1,
                damp=0.01, scale=25, recursion_depth=5000, r=1, start=0):
    """Calculates s_test for the whole test dataset taking into account all
    training data images.

    Arguments:
        model: pytorch model, for which s_test should be calculated
        test_loader: pytorch dataloader, which can load the test data
        train_loader: pytorch dataloader, which can load the train data
        save: Path, path where to save the s_test files if desired. Omitting
            this argument will skip saving
        gpu: int, device id to use for GPU, -1 for CPU (default)
        damp: float, influence function damping factor
        scale: float, influence calculation scaling factor
        recursion_depth: int, number of recursions to perform during s_test
            calculation, increases accuracy. r*recursion_depth should equal the
            training dataset size.
        r: int, number of iterations of which to take the avg.
            of the h_estimate calculation; r*recursion_depth should equal the
            training dataset size.
        start: int, index of the first test index to use. default is 0

    Returns:
        s_tests: list of torch vectors, contain all s_test for the whole
            dataset. Can be huge.
        save: Path, path to the folder where the s_test files were saved to or
            False if they were not saved."""
    if save and not isinstance(save, Path):
        save = Path(save)
    if not save:
        logging.info("ATTENTION: not saving s_test files.")

    s_tests = []
    for i in range(start, len(test_loader.dataset)):
        z_test, t_test = test_loader.dataset[i]
        z_test = test_loader.collate_fn([z_test])
        t_test = test_loader.collate_fn([t_test])

        s_test_vec = calc_s_test_single(model, z_test, t_test, z_test2, t_test2, train_loader,
                                        gpu, damp, scale, recursion_depth, r)

        if save:
            s_test_vec = [s.cpu() for s in s_test_vec]
            torch.save(
                s_test_vec,
                save.joinpath(f"{i}_recdep{recursion_depth}_r{r}.s_test"))
        else:
            s_tests.append(s_test_vec)
        display_progress(
            "Calc. z_test (s_test): ", i-start, len(test_loader.dataset)-start)

    return s_tests, save


def calc_s_test_single(w, X_test_sample, y_test_sample, X_train, y_train, gpu=-1,
                       damp=0.01, scale=25, recursion_depth=5000, r=1):
    """Calculates s_test for a single test image taking into account the whole
    training dataset. s_test = invHessian * nabla(Loss(test_img, model params))

    Arguments:
        model: pytorch model, for which s_test should be calculated
        z_test: test image
        t_test: test image label
        train_loader: pytorch dataloader, which can load the train data
        gpu: int, device id to use for GPU, -1 for CPU (default)
        damp: float, influence function damping factor
        scale: float, influence calculation scaling factor
        recursion_depth: int, number of recursions to perform during s_test
            calculation, increases accuracy. r*recursion_depth should equal the
            training dataset size.
        r: int, number of iterations of which to take the avg.
            of the h_estimate calculation; r*recursion_depth should equal the
            training dataset size.

    Returns:
        s_test_vec: torch tensor, contains s_test for a single test image"""
    s_test_vec_list = []
    for i in range(r):
        s_test_vec_list.append(s_test(X_test_sample, y_test_sample, w, X_train, y_train,
                                      gpu=gpu, damp=damp, scale=scale,
                                      recursion_depth=recursion_depth))
        display_progress("Averaging r-times: ", i, r)

    ################################
    # TODO: Understand why the first[0] tensor is the largest with 1675 tensor
    #       entries while all subsequent ones only have 335 entries?
    ################################
    s_test_vec = s_test_vec_list[0]
    for i in range(1, r):
        s_test_vec += s_test_vec_list[i]

    s_test_vec = [i / r for i in s_test_vec]

    return s_test_vec


def calc_grad_z(model, train_loader, save_pth=False, gpu=-1, start=0):
    """Calculates grad_z and can save the output to files. One grad_z should
    be computed for each training data sample.

    Arguments:
        model: pytorch model, for which s_test should be calculated
        train_loader: pytorch dataloader, which can load the train data
        save_pth: Path, path where to save the grad_z files if desired.
            Omitting this argument will skip saving
        gpu: int, device id to use for GPU, -1 for CPU (default)
        start: int, index of the first test index to use. default is 0

    Returns:
        grad_zs: list of torch tensors, contains the grad_z tensors
        save_pth: Path, path where grad_z files were saved to or
            False if they were not saved."""
    if save_pth and isinstance(save_pth, str):
        save_pth = Path(save_pth)
    if not save_pth:
        logging.info("ATTENTION: Not saving grad_z files!")

    grad_zs = []
    for i in range(start, len(train_loader.dataset)):
        z, t = train_loader.dataset[i]
        z = train_loader.collate_fn([z])
        t = train_loader.collate_fn([t])
        grad_z_vec = grad_z(z, t, model, gpu=gpu)
        if save_pth:
            grad_z_vec = [g.cpu() for g in grad_z_vec]
            torch.save(grad_z_vec, save_pth.joinpath(f"{i}.grad_z"))
        else:
            grad_zs.append(grad_z_vec)
        display_progress(
            "Calc. grad_z: ", i-start, len(train_loader.dataset)-start)

    return grad_zs, save_pth


def load_s_test(s_test_dir=Path("./s_test/"), s_test_id=0, r_sample_size=10,
                train_dataset_size=-1):
    """Loads all s_test data required to calculate the influence function
    and returns a list of it.

    Arguments:
        s_test_dir: Path, folder containing files storing the s_test values
        s_test_id: int, number of the test data sample s_test was calculated
            for
        r_sample_size: int, number of s_tests precalculated
            per test dataset point
        train_dataset_size: int, number of total samples in dataset;
            -1 indicates to use all available grad_z files

    Returns:
        e_s_test: list of torch vectors, contains all e_s_tests for the whole
            dataset.
        s_test: list of torch vectors, contain all s_test for the whole
            dataset. Can be huge."""
    if isinstance(s_test_dir, str):
        s_test_dir = Path(s_test_dir)

    s_test = []
    logging.info(f"Loading s_test from: {s_test_dir} ...")
    num_s_test_files = len(s_test_dir.glob("*.s_test"))
    if num_s_test_files != r_sample_size:
        logging.warn("Load Influence Data: number of s_test sample files"
                     " mismatches the available samples")
    ########################
    # TODO: should prob. not hardcode the file name, use natsort+glob
    ########################
    for i in range(num_s_test_files):
        s_test.append(
            torch.load(s_test_dir / str(s_test_id) + f"_{i}.s_test"))
        display_progress("s_test files loaded: ", i, r_sample_size)

    #########################
    # TODO: figure out/change why here element 0 is chosen by default
    #########################
    e_s_test = s_test[0]
    # Calculate the sum
    for i in range(len(s_test)):
        e_s_test = [i + j for i, j in zip(e_s_test, s_test[0])]

    # Calculate the average
    #########################
    # TODO: figure out over what to calculate the average
    #       should either be r_sample_size OR e_s_test
    #########################
    e_s_test = [i / len(s_test) for i in e_s_test]

    return e_s_test, s_test


def load_grad_z(grad_z_dir=Path("./grad_z/"), train_dataset_size=-1):
    """Loads all grad_z data required to calculate the influence function and
    returns it.

    Arguments:
        grad_z_dir: Path, folder containing files storing the grad_z values
        train_dataset_size: int, number of total samples in dataset;
            -1 indicates to use all available grad_z files

    Returns:
        grad_z_vecs: list of torch tensors, contains the grad_z tensors"""
    if isinstance(grad_z_dir, str):
        grad_z_dir = Path(grad_z_dir)

    grad_z_vecs = []
    logging.info(f"Loading grad_z from: {grad_z_dir} ...")
    available_grad_z_files = len(grad_z_dir.glob("*.grad_z"))
    if available_grad_z_files != train_dataset_size:
        logging.warn("Load Influence Data: number of grad_z files mismatches"
                     " the dataset size")
        if -1 == train_dataset_size:
            train_dataset_size = available_grad_z_files
    for i in range(train_dataset_size):
        grad_z_vecs.append(torch.load(grad_z_dir / str(i) + ".grad_z"))
        display_progress("grad_z files loaded: ", i, train_dataset_size)

    return grad_z_vecs


def calc_influence_function(train_dataset_size, grad_z_vecs=None,
                            e_s_test=None):
    """Calculates the influence function

    Arguments:
        train_dataset_size: int, total train dataset size
        grad_z_vecs: list of torch tensor, containing the gradients
            from model parameters to loss
        e_s_test: list of torch tensor, contains s_test vectors

    Returns:
        influence: list of float, influences of all training data samples
            for one test sample
        harmful: list of float, influences sorted by harmfulness
        helpful: list of float, influences sorted by helpfulness"""
    if not grad_z_vecs and not e_s_test:
        grad_z_vecs = load_grad_z()
        e_s_test, _ = load_s_test(train_dataset_size=train_dataset_size)

    if (len(grad_z_vecs) != train_dataset_size):
        logging.warn("Training data size and the number of grad_z files are"
                     " inconsistent.")
        train_dataset_size = len(grad_z_vecs)

    influences = []
    for i in range(train_dataset_size):
        tmp_influence = -sum(
            [
                ###################################
                # TODO: verify if computation really needs to be done
                # on the CPU or if GPU would work, too
                ###################################
                torch.sum(k * j).data.cpu().numpy()
                for k, j in zip(grad_z_vecs[i], e_s_test)
                ###################################
                # Originally with [i] because each grad_z contained
                # a list of tensors as long as e_s_test list
                # There is one grad_z per training data sample
                ###################################
            ]) / train_dataset_size
        influencex = [
                ###################################
                # TODO: verify if computation really needs to be done
                # on the CPU or if GPU would work, too
                ###################################
                torch.sum(k * j).data.cpu().numpy()
                for k, j in zip(grad_z_vecs[i], e_s_test)
                ###################################
                # Originally with [i] because each grad_z contained
                # a list of tensors as long as e_s_test list
                # There is one grad_z per training data sample
                ###################################
            ]
        print(len(influencex))
        exit()
        influences.append(tmp_influence)
        display_progress("Calc. influence function: ", i, train_dataset_size)

    harmful = np.argsort(influences)
    helpful = harmful[::-1]

    return influences, harmful.tolist(), helpful.tolist()


def calc_influence_single(w, X_train, y_train, X_test_sample, y_test_sample, gpu,
                          recursion_depth, r, s_test_vec=None,
                          time_logging=False):

    if not s_test_vec:
        s_test_vec = calc_s_test_single(w, X_test_sample, y_test_sample, X_train, y_train,
                                        gpu, recursion_depth=recursion_depth,
                                        r=r)

    # Calculate the influence function
    #print(s_test_vec.size())
    train_dataset_size = len(train_loader.dataset)
    influences = []
    for i in range(train_dataset_size):
    #for i in range(50):
        _, z, t, _, _ = train_loader.dataset[i]
        z = train_loader.collate_fn([z])
        t = train_loader.collate_fn([t])
        if time_logging:
            time_a = datetime.datetime.now()
        grad_z_vec = grad_z(z, t, model, gpu=gpu)
        if time_logging:
            time_b = datetime.datetime.now()
            time_delta = time_b - time_a
            logging.info(f"Time for grad_z iter:"
                         f" {time_delta.total_seconds() * 1000}")
        #print(z.size())
        #print(t.size())
        #print(type(grad_z_vec))
        #print(type(s_test_vec))
        #print(len(grad_z_vec))
        #print(grad_z_vec[0].size())
        #print(len(s_test_vec))
        #print(s_test_vec[0].size())
        #print(torch.sum(grad_z_vec[0] * s_test_vec[0]).data)
        #print((grad_z_vec[0] * s_test_vec[0]).size())
        #exit() 
        
        tmp_influence = -sum(
            [
                ####################
                # TODO: potential bottle neck, takes 17% execution time
                # torch.sum(k * j).data.cpu().numpy()
                ####################
                torch.sum(k * j).data
                for k, j in zip(grad_z_vec, s_test_vec)
            ]) / train_dataset_size
        
        influences.append(tmp_influence.item())
        display_progress("Calc. influence function: ", i, train_dataset_size)

    harmful = np.argsort(influences)
    helpful = harmful[::-1]
    #print(influences)
    return influences, harmful.tolist(), helpful.tolist(), test_id_num



def get_dataset_sample_ids_per_class(class_id, num_samples, test_loader,
                                     start_index=0):
    """Gets the first num_samples from class class_id starting from
    start_index. Returns a list with the indicies which can be passed to
    test_loader.dataset[X] to retreive the actual data.

    Arguments:
        class_id: int, name or id of the class label
        num_samples: int, number of samples per class to process
        test_loader: DataLoader, can load the test dataset.
        start_index: int, means after which x occourance to add an index
            to the list of indicies. E.g. if =3, then it would add the
            4th occourance of an item with the label class_nr to the list.

    Returns:
        sample_list: list of int, contains indicies of the relevant samples"""
    #######
    # NOTE / TODO: here's optimisation potential. We are currently searching
    # for the x+1th sample and when that's found we cancel the loop. we could
    # stop after finding the x'th picture (start_index + num_samples)
    #######
    sample_list = []
    name_list = []
    img_count = 0
    #print(len(test_loader.dataset))
    for i in range(len(test_loader.dataset)):
        #print(test_loader.dataset[i])
        _, _, t, _, name = test_loader.dataset[i]
        if class_id == t:
            img_count += 1
            if (img_count > start_index) and \
                    (img_count <= start_index + num_samples):
                sample_list.append(i)
                name_list.append(name)
            elif img_count > start_index + num_samples:
                break

    return sample_list, name_list


def get_dataset_sample_ids(num_samples, test_loader, num_classes=None,
                           start_index=0):
    """Gets the first num_sample indices of all classes starting from
    start_index per class. Returns a list and a dict containing the indicies.

    Arguments:
        num_samples: int, number of samples of each class to return
        test_loader: DataLoader, can load the test dataset
        num_classes: int, number of classes contained in the dataset
        start_index: int, means after which x occourance to add an index
            to the list of indicies. E.g. if =3, then it would add the
            4th occourance of an item with the label class_nr to the list.

    Returns:
        sample_dict: dict, containing dict[class] = list_of_indices
        sample_list: list, containing a continious list of indices"""
    sample_dict = {}
    name_dict = {}
    sample_list = []
    name_list = []
    if not num_classes:
        num_classes = len(np.unique(test_loader.dataset.targets))
    if start_index > len(test_loader.dataset) / num_classes:
        logging.warn(f"The variable test_start_index={start_index} is "
                     f"larger than the number of available samples per class.")
    for i in range(num_classes):
        sample_dict[str(i)], name_dict[str(i)] = get_dataset_sample_ids_per_class(
            i, num_samples, test_loader, start_index)
        # Append the new list on the same level as the old list
        # Avoids having a list of lists
        sample_list[len(sample_list):len(sample_list)] = sample_dict[str(i)]
        
        name_list[len(name_list):len(name_list)] = name_dict[str(i)]
    return sample_dict, sample_list, name_list


def calc_img_wise(config, w, X_train, y_train, X_test, y_test, test_dataset_iter_len=10):
    

    influences_meta = copy.deepcopy(config)
    test_sample_num = config['test_sample_num']
    test_start_index = config['test_start_index']
    outdir = Path(config['outdir'])
    outdir.mkdir(exist_ok=True, parents=True)

    test_dataset_iter_len = test_dataset_iter_len
    sample_list = list(range(test_dataset_iter_len))
    #print(name_list)
    # Set up logging and save the metadata conf file
    logging.info(f"Running on: {test_sample_num} images per class.")
    logging.info(f"Starting at img number: {test_start_index} per class.")
    influences_meta['test_sample_index_list'] = sample_list
    influences_meta_fn = f"influences_results_meta_{test_start_index}-" \
                         f"{test_sample_num}.json"
    influences_meta_path = outdir.joinpath(influences_meta_fn)
    save_json(influences_meta, influences_meta_path)

    influences = {}
    # Main loop for calculating the influence function one test sample per
    # iteration.
    path = '/home/ruizhe/ModelDebias/Baselines/Learning-Debiased-Disentangled-master/data/influence2/'
    

    
    harmfuls = []
    
    for j in range(0, test_dataset_iter_len):
        # If we calculate evenly per class, choose the test img indicies
        # from the sample_list instead
        i = j
        
        logging.info(f"number: {i} ")
        print(i)
        start_time = time.time()
        influence, harmful, helpful, _ = calc_influence_single(
            w, X_train, y_train, X_test[i], y_test[i], gpu=config['gpu'],
            recursion_depth=config['recursion_depth'], r=config['r_averaging'])
        #print(influence)
        end_time = time.time()
        
        harmfuls.extend(harmful[:50])
        

        ###########
        # Different from `influence` above
        ###########

        #avg_infl.append(influence)

        '''
        harmful_list = []
        for k in harmful[:7]:
            harmful_list.append(train_loader.dataset[k][4])
            
        helpful_list = []
        for k in helpful[:7]:
            helpful_list.append(train_loader.dataset[k][4])
            
        path2 = path + str(i) + '/'
        if not os.path.exists(path2):
            os.mkdir(path2)
        
        shutil.copyfile(test_loader.dataset[i][4], path2 + test_loader.dataset[i][4].split('/')[-1])
        print("test_sample:", test_loader.dataset[i][4])
        logging.info(f"test_sample: {test_loader.dataset[i][4]} ")
        
        for sample in harmful_list:
            shutil.copyfile(sample, path2 + sample.split('/')[-1])
        
        for sample in helpful_list:
            shutil.copyfile(sample, path2 + sample.split('/')[-1])
        
        print("harmful_list:", harmful_list)
        print("helpful_list:", helpful_list)
        
        logging.info(f"harmful_list: {harmful_list} ")
        logging.info(f"helpful_list: {helpful_list} ")
        
        save_json(influences, tmp_influences_path)
        '''
        #display_progress("Test samples processed: ", j, test_dataset_iter_len)
    '''        
    avg_infl = np.array(avg_infl)
    avg_infl = np.mean(avg_infl, axis=0)
    #print(avg_infl)
    avg_infl = avg_infl.tolist()
    harmful = np.argsort(avg_infl)
    helpful = harmful[::-1]
    
    harmful = harmful[:500]
    helpful = helpful[:500]
    '''


    


    #influences_path = outdir.joinpath(f"influence_results_{test_start_index}_"f"{test_sample_num}.json")
    #save_json(influences, influences_path)

    return avg_infl, harmfuls, helpful


def calc_all_grad_then_test(config, model, train_loader, test_loader):
    """Calculates the influence function by first calculating
    all grad_z, all s_test and then loading them to calc the influence"""

    outdir = Path(config['outdir'])
    outdir.mkdir(exist_ok=True, parents=True)

    s_test_outdir = outdir.joinpath("s_test/")
    if not s_test_outdir.exists():
        s_test_outdir.mkdir()
    grad_z_outdir = outdir.joinpath("grad_z/")
    if not grad_z_outdir.exists():
        grad_z_outdir.mkdir()

    influence_results = {}

    calc_s_test(model, test_loader, train_loader, s_test_outdir,
                config['gpu'], config['damp'], config['scale'],
                config['recursion_depth'], config['r_averaging'],
                config['test_start_index'])
    calc_grad_z(model, train_loader, grad_z_outdir, config['gpu'],
                config['test_start_index'])

    train_dataset_len = len(train_loader.dataset)
    influences, harmful, helpful = calc_influence_function(train_dataset_len)

    influence_results['influences'] = influences
    influence_results['harmful'] = harmful
    influence_results['helpful'] = helpful
    influences_path = outdir.joinpath("influence_results.json")
    save_json(influence_results, influences_path)

    
def calc_train_img_wise(config, model, train_loader, test_loader, gpu=-1, damp=0.01, scale=25.0,
           recursion_depth=5000):
    
    for i in range(len(train_loader.dataset)):
        #print(test_loader.dataset[i])
        _, z_train, t_train, _, name = train_loader.dataset[i]
        
        z_train = train_loader.collate_fn([z_train])
        t_train = train_loader.collate_fn([t_train])
        s_test_vec = s_test(z_train, t_train, model, train_loader,
                                      gpu=config['gpu'],
            recursion_depth=config['recursion_depth'], damp=0.01, scale=25.0)
        print(s_test_vec[0].size())
        exit()


    return s_test_vec


