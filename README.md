# Re-ZERO
2023-1 SKKU AI S.E.L.F. Re:ZERO

## CIFAR10
```bash
$ python cifar_train_backbone.py --arch resnet32 /
                                 --dataset cifar10 --data_path './dataset/data_img' /
                                 --loss_type BSCE --imb_factor 0.01 /
                                 --batch_size 64 --learning_rate 0.1 
```

## CIFAR100
```bash
$ python cifar_train_backbone.py --arch resnet32 /
                                 --dataset cifar100 --data_path './dataset/data_img' /
                                 --loss_type BSCE --imb_factor 0.01 /
                                 --batch_size 64 --learning_rate 0.1 
```

## Notice

*Note: I have modified the code several times. The code in this repository may need to be modified.

## Other Resources of long-tailed visual recognition
[Awesome-LongTailed-Learning](https://github.com/Vanint/Awesome-LongTailed-Learning)

[Awesome-of-Long-Tailed-Recognition](https://github.com/zwzhang121/Awesome-of-Long-Tailed-Recognition)

[Long-Tailed-Classification-Leaderboard](https://github.com/yanyanSann/Long-Tailed-Classification-Leaderboard)

[BagofTricks-LT](https://github.com/zhangyongshun/BagofTricks-LT)

## Acknowledgment
We refer to some codes from [GCL](https://github.com/Keke921/GCLLoss). Many thanks to the authors.
