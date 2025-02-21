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
}

for genus in ground_truths.keys():

    x = list(ground_truths[genus].keys())
    y = list(ground_truths[genus].values())
    fontsize = 18

    plt.figure(figsize=(20, 10))
    plt.bar(x, y, width=5, edgecolor='white')
    plt.xlim(3760, 3940)
    plt.xticks(np.arange(3760, 3940, 20), rotation=45, fontsize=fontsize)
    plt.ylim(0, 60)
    plt.yticks(np.arange(0, 60, 10), fontsize=fontsize)
    plt.xlabel("Depth", fontsize=18)
    plt.ylabel("Count", fontsize=18)
    plt.tight_layout()
    plt.savefig(f"{genus}_ground_truth_distribution.png", dpi=300)
    plt.close()