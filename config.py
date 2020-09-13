class DefaultConfig(object):
    env='default'
    model='ResNet34'
    # train_data_root='./dataset/training_set'
    # test_data_root = './dataset/test_set'
    train_data_root='./data/train_imgs'
    test_data_root = './data/test_imgs'
    #load_model_path='checkpoints/model.pth'
    load_model_path = None

    batch_size=8
    use_gpu=True
    num_workers=4
    print_freq=20

    debug_file='/tmp/debug'
    result_file='result.csv'

    max_epoch=100
    lr=0.001
    lr_decay=0.95
    weight_decay=1e-4