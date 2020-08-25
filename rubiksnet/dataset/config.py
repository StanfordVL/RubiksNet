import os


def return_ucf101(root_path):
    filename_categories = 101
    root_data = os.path.join(root_path, "ucf101/rgb")
    filename_imglist_train = "ucf101/label/train.txt"
    filename_imglist_val = "ucf101/label/val.txt"
    prefix = "img_{:05d}.jpg"
    return (
        filename_categories,
        filename_imglist_train,
        filename_imglist_val,
        root_data,
        prefix,
    )


def return_hmdb51(root_path):
    filename_categories = 51
    root_data = os.path.join(root_path, "hmdb/rgb")
    filename_imglist_train = "hmdb/label/train.txt"
    filename_imglist_val = "hmdb/label/val.txt"
    prefix = "img_{:05d}.jpg"
    return (
        filename_categories,
        filename_imglist_train,
        filename_imglist_val,
        root_data,
        prefix,
    )


def return_somethingv1(root_path):
    filename_categories = "somethingv1/label/category.txt"
    root_data = os.path.join(root_path, "somethingv1/rgb")
    filename_imglist_train = "somethingv1/label/train_videofolder.txt"
    filename_imglist_val = "somethingv1/label/val_videofolder.txt"
    prefix = "{:05d}.jpg"
    return (
        filename_categories,
        filename_imglist_train,
        filename_imglist_val,
        root_data,
        prefix,
    )


def return_somethingv2(root_path):
    filename_categories = "somethingv2/label/category.txt"
    filename_imglist_train = "somethingv2/label/train_videofolder.txt"
    root_data = os.path.join(root_path, "somethingv2/rgb")
    filename_imglist_val = "somethingv2/label/val_videofolder.txt"
    prefix = "{:06d}.jpg"
    return (
        filename_categories,
        filename_imglist_train,
        filename_imglist_val,
        root_data,
        prefix,
    )


def return_kinetics(root_path):
    filename_categories = 400
    root_data = os.path.join(root_path, "kinetics/images")
    filename_imglist_train = "kinetics/labels/train_videofolder.txt"
    filename_imglist_val = "kinetics/labels/val_videofolder.txt"
    prefix = "img_{:05d}.jpg"
    return (
        filename_categories,
        filename_imglist_train,
        filename_imglist_val,
        root_data,
        prefix,
    )


def return_dataset(dataset, root_path):
    dict_single = {
        "something": return_somethingv2,
        "somethingv2": return_somethingv2,
        "somethingv1": return_somethingv1,
        "ucf101": return_ucf101,
        "hmdb": return_hmdb51,
        "kinetics": return_kinetics,
    }
    if dataset in dict_single:
        (
            file_categories,
            file_imglist_train,
            file_imglist_val,
            root_data,
            prefix,
        ) = dict_single[dataset](root_path)
    else:
        raise ValueError("Unknown dataset " + dataset)

    file_imglist_train = os.path.join(root_path, file_imglist_train)
    file_imglist_val = os.path.join(root_path, file_imglist_val)
    if isinstance(file_categories, str):
        file_categories = os.path.join(root_path, file_categories)
        with open(file_categories) as f:
            lines = f.readlines()
        categories = [item.rstrip() for item in lines]
    else:
        categories = [None] * file_categories
    n_class = len(categories)
    # print('{}: {} classes'.format(dataset, n_class))
    return n_class, file_imglist_train, file_imglist_val, root_data, prefix
