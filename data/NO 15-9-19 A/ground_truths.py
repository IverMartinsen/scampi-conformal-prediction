import numpy as np
import matplotlib.pyplot as plt

ground_truths = {
    'dissiliodinium':
        {
            3834: 1,
            3846: 1,
        },
    'rigaudella':
        {
            3780: 42,
            3786: 49,
            3792: 56,
            3798: 43,
            3804: 30,
            3810: 52,
            3816: 41,
            3822: 21,
            3828: 2,
            3834: 4,
        },
    'sirmiodinium':
        {
            3780: 3,
            3786: 4,
            3792: 3,
            3798: 1,
            3822: 1,
            3876: 2,
            3891: 3,
        },
    'surculosphaeridium':
        {
            3798: 18,
            3804: 7,
            3810: 6,
            3816: 1,
        },
    'bisaccate':
        {
            3780: 18,
            3786: 21,
            3792: 16,
            3798: 17,
            3804: 18,
            3810: 29,
            3816: 18,
            3822: 27,
            3824: 85,
            3830: 76,
            3834: 68,
            3835: 27,
            3837: 29,
            3839: 19,
            3841: 23,
            3844: 22,
            3846: 18,
            3848: 14,
            3851: 20,
            3854: 21,
            3870: 27,
            3876: 24,
            3891: 22,
            3894: 19,
            3898: 10,
            3908: 18,
            3919: 19,
        },
}

for genus in ground_truths.keys():

    x = list(ground_truths[genus].keys())
    y = list(ground_truths[genus].values())
    fontsize = 18

    plt.figure(figsize=(20, 10))
    plt.bar(x, y, width=10, edgecolor='white')
    plt.xlim(3760, 3940)
    plt.xticks(np.arange(3760, 3940, 20), rotation=45, fontsize=fontsize)
    plt.ylim(0, 60)
    plt.yticks(np.arange(0, 60, 10), fontsize=fontsize)
    if genus == 'bisaccate':
        plt.ylim(0, 100)
        plt.yticks(np.arange(0, 100, 20), fontsize=fontsize)
    plt.xlabel("Depth", fontsize=18)
    plt.ylabel("Count", fontsize=18)
    plt.tight_layout()
    plt.savefig(f"{genus}_ground_truth_distribution.png", dpi=300)
    plt.close()