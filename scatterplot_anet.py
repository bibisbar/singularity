import matplotlib.pyplot as plt
import json
import os
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

# import data#13 14

with open('/home/wiss/zhang/nfs/reb_deltap/results_sin/anet_reb_beta_0.25_lr_0.0001_anet_of_bin_13_delta_p.json') as f:
    data = json.load(f)
    
with open('/home/wiss/zhang/Yanyu/infovis/recall/recall_sin_anet_delta_reb.json') as f:
    data_ori = json.load(f)


category_colors = {'temporal_contact_swap': 'blue', 'temporal_action_swap': 'orange', 'neighborhood_same_entity': 'green', 'neighborhood_diff_entity': 'red', 'counter_spatial': 'purple', 'counter_action': 'brown', 'counter_contact': 'pink', 'counter_attribute': 'grey'}
# legend_elements = [Line2D([1], [1], marker='o', color='w', markerfacecolor='black', markersize=14, label='Clip4clip(mean)'),
#                     Line2D([1], [1], marker='s', color='w', markerfacecolor='black', markersize=14, label='Clip4clip(transf)'),
#                     Line2D([1], [1], marker='^', color='w', markerfacecolor='black', markersize=14, label='Singularity-temporal'),
#                     Line2D([1], [1], marker='d', color='w', markerfacecolor='black', markersize=14, label='X-CLIP'),
#                     Line2D([1], [1], marker='v', color='w', markerfacecolor='black', markersize=14, label='ALPRO')]
legend_elements = [Line2D([1], [1], marker='o', color='w', markerfacecolor='black', markersize=14, label='before post-training'),
                    Line2D([1], [1], marker='^', color='w', markerfacecolor='black', markersize=14, label='after post-training')]
# for category, values in data.items():
#     plt.scatter(values['t2v'][0], values['v2t'][0], label=category, marker='^', s=200, alpha=0.6, color=category_colors.get(category, 'black'))
    
# for category, values in data_ori.items():
#     plt.scatter(values['t2v'][0], values['v2t'][0], label=category, marker='o', s=200, alpha=0.6, color=category_colors.get(category, 'black'))
   

# for category, values in data_ori.items():
#     plt.annotate(
#         "",
#         xy=(data[category]['t2v'][0], data[category]['v2t'][0]),
#         xytext=(values['t2v'][0], values['v2t'][0]),
#         arrowprops=dict(
#             color=category_colors.get(category, 'black'),
#             linestyle='dashed',
#             width=0.01,
#             headwidth=7,  # Arrowhead width
#             headlength=7,  # Arrowhead length
#             shrinkA=0,
#             shrinkB=0,
#             patchA=None,
#             patchB=None,
#         ),  
#     )
            
    
for category, values in data.items():
    plt.scatter(values['t2v'][1], values['v2t'][1], label=category, marker='^', s=200, alpha=0.6, color=category_colors.get(category, 'black'))
    
for category, values in data_ori.items():
    plt.scatter(values['t2v'][1], values['v2t'][1], label=category, marker='o', s=200, alpha=0.6, color=category_colors.get(category, 'black'))
 
for category, values in data_ori.items():
    plt.annotate(
        "",
        xy=(data[category]['t2v'][1], data[category]['v2t'][1]),
        xytext=(values['t2v'][1], values['v2t'][1]),
        arrowprops=dict(
            color=category_colors.get(category, 'black'),
            linestyle='dashed',
            width=0.01,
            headwidth=7,  # Arrowhead width
            headlength=7,  # Arrowhead length
            shrinkA=0,
            shrinkB=0,
            patchA=None,
            patchB=None,
        ),  
    )
# plt.xlim(-0.9, 0.9) #recall1
# plt.ylim(-0.9, 0.9)
# plt.xlim(-0.8, 0.8) #recall1
# plt.ylim(-0.8, 0.8)
plt.xlim(-0.6, 0.6)  #recall5
plt.ylim(-0.6, 0.6)
plt.subplots_adjust(top=0.75)
plt.gcf().set_size_inches(8, 6)
plt.gca().xaxis.set_major_locator(MultipleLocator(0.2))
plt.gca().yaxis.set_major_locator(MultipleLocator(0.2))
plt.xlabel(r'$\Delta p_{R@5}$' + ' on Text-to-Video', fontsize=16)
plt.ylabel(r'$\Delta p_{R@5}$' + ' on Video-to-Text', fontsize=16)
legend1 = plt.legend(['temporal contact_swap', 'temporal action_swap', 'temporal attribute', 'neighborhood attribute', 'counterfactual spatial relationship','counterfactual action','counterfactual contact', 'counterfactual attribute'], bbox_to_anchor=(0.5, 1.4), loc='upper center', ncol=2, fontsize=14)
legend2 = plt.legend(loc='lower center', handles = legend_elements, ncol =3, fontsize=16)
plt.gca().add_artist(legend1)
plt.gca().add_artist(legend2)
plt.show()


save_path = '/home/wiss/zhang/nfs/reb_deltap/scatter_anet'

if not os.path.exists(save_path):
    os.makedirs(save_path)
    
plt.savefig(save_path + '/anet_reb_beta_0.25_lr_0.0001_sin__r5_13' + '.pdf')
plt.savefig(save_path + '/anet_reb_beta_0.25_lr_0.0001_sin__r5_13' + '.png')
plt.close()
#__r1  __r5