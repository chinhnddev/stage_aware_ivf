from ivf.data.transforms import assert_no_augmentation, get_eval_transforms, get_train_transforms, has_augmentation


def test_eval_transforms_have_no_augmentation():
    eval_tf = get_eval_transforms()
    assert_no_augmentation(eval_tf)
    assert not has_augmentation(eval_tf)


def test_train_transforms_have_augmentation():
    train_tf = get_train_transforms("light")
    assert has_augmentation(train_tf)
