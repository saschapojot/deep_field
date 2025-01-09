import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
inCoefDir="./coefs/"

in_S0_file=inCoefDir+"/S0.pth"
width=4
height=4
S0=torch.load(in_S0_file)

S0_x=np.array(S0[0,0,:,:])
S0_y=np.array(S0[0,1,:,:])
S0_z=np.array(S0[0,2,:,:])
###########################################################
#plot S0_x
plt.figure(figsize=(width, height))
plt.imshow(S0_x,cmap='gray', interpolation='nearest')
# Remove x and y ticks
plt.axis('off')  # Turn off the axes
plt.tight_layout()
plt.savefig(inCoefDir+"S0_x.svg",format="svg", bbox_inches='tight', pad_inches=0)
plt.close()
###########################################################


###########################################################
#plot Phi0 (this is T0 in paper)
cmaps_vec=["Reds","Oranges","Greens","Blues","Purples"]
in_Phi0_file=inCoefDir+"/Phi0.pth"
out_Phi0_dir=inCoefDir+"/Phi0Pics/"
Path(out_Phi0_dir).mkdir(parents=True, exist_ok=True)
Phi0=torch.load(in_Phi0_file)
C_num_to_show=5
print(f"Phi0.shape={Phi0.shape}")
Phi0_to_plt=Phi0[0,0:C_num_to_show,:,:].detach().numpy()

##plot Phi0, channel0
# Phi0_channel0=Phi0_to_plt[0,:,:]
# plt.figure(figsize=(width, height))
# plt.imshow(Phi0_channel0, cmap=cmaps_vec[0], interpolation='nearest')  # Use the Reds colormap
# plt.axis('off')  # Turn off the axes
# plt.tight_layout()
# plt.savefig(out_Phi0_dir+"/Phi0_channel0.svg",format="svg", bbox_inches='tight', pad_inches=0)
# plt.close()
for j in range(0,C_num_to_show):
    plt.figure(figsize=(width, height))
    Phi0_channel_j=Phi0_to_plt[j,:,:]
    plt.imshow(Phi0_channel_j,cmap=cmaps_vec[j],interpolation='nearest')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_Phi0_dir + f"/Phi0_channel{j}.svg", format="svg", bbox_inches='tight', pad_inches=0)
    plt.close()

    #colorbar
    fig, ax = plt.subplots()  # Adjust the aspect ratio
    norm = plt.Normalize(vmin=Phi0_channel_j.min(), vmax=Phi0_channel_j.max())
    cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmaps_vec[j]),  # Use the 'Reds' colormap
        cax=ax,
        orientation='vertical'
    )
    cb.set_label('Value', fontsize=12)  # Add a label if desired
    plt.savefig(out_Phi0_dir+f"/colorbar_Phi0_channel{j}.svg")
    plt.close()
    ## plot Phi0, channel1