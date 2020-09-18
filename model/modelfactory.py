
def get_model(args, classes):
    channels = 256
    if args.dataset == "omniglot":
        in_ch = 1
        last_conv_layer_kernel = 1
        n = 1
    elif args.dataset == "imagenet" or args.dataset == "cub" or args.dataset == "cifar":
        in_ch = 3
        last_conv_layer_kernel = 3
        n = 9
    else:
        print("Unsupported model; either implement the model in model/ModelFactory or choose a different model")
        assert (False)

    config_net = [
        {"name": 'conv2d', "adaptation": False, "meta": True,
         "config": {"out-channels": channels, "in-channels": in_ch, "kernal": 3, "stride": 2, "padding": 0}},
        {"name": 'relu'},

        {"name": 'conv2d', "adaptation": False, "meta": True,
         "config": {"out-channels": channels, "in-channels": channels, "kernal": 3, "stride": 1,
                    "padding": 0}},
        {"name": 'relu'},

        {"name": 'conv2d', "adaptation": False, "meta": True,
         "config": {"out-channels": channels, "in-channels": channels, "kernal": 3, "stride": 2,
                    "padding": 0}},
        {"name": 'relu'},
        #
        {"name": 'conv2d', "adaptation": False, "meta": True,
         "config": {"out-channels": channels, "in-channels": channels, "kernal": 3, "stride": 1,
                    "padding": 0}},
        {"name": 'relu'},

        {"name": 'conv2d', "adaptation": False, "meta": True,
         "config": {"out-channels": channels, "in-channels": channels, "kernal": 3, "stride": 2,
                    "padding": 0}},
        {"name": 'relu'},

        {"name": 'conv2d', "adaptation": False, "meta": True,
         "config": {"out-channels": channels, "in-channels": channels, "kernal": last_conv_layer_kernel, "stride": 2,
                    "padding": 0}},
        {"name": 'relu'},

        {"name": 'flatten'},

        {"name": 'linear', "adaptation": True, "meta": True,
         "config": {"out-channels": 1024, "in-channels": n * channels}},
        {"name": 'linear', "adaptation": True, "meta": True,
         "config": {"out-channels": classes, "in-channels": 1024}}
    ]

    if args.attention:
        config_net.insert(13, {"name": 'linear', "adaptation": True, "meta": True,
                               "config": {"out-channels": n * channels, "in-channels": n * channels}})
        config_net.insert(14, {"name": 'tanh'},)
        config_net.insert(15, {"name": 'linear', "adaptation": True, "meta": True,
                               "config": {"out-channels": 1, "in-channels": n * channels}})
        config_net.insert(16, {"name": 'softmax'})
        config_net.insert(17, {"name": 'sum'})
    elif args.mean:
        config_net.insert(13, {"name": 'mean'})

    return config_net


