import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
inCoefDir="./coefs/"
#this script plots layers

width=4
height=4

outCoefDir="../fig_coefs/"
###########################################################
#plot S0_x
in_S0_file=inCoefDir+"/S0.pth"
out_S0_dir=outCoefDir+"/S0Pics/"
Path(out_S0_dir).mkdir(parents=True, exist_ok=True)
S0=torch.load(in_S0_file)

S0_x=np.array(S0[0,0,:,:])
S0_y=np.array(S0[0,1,:,:])
S0_z=np.array(S0[0,2,:,:])
S0_to_plt=S0[0,:,:,:].detach().numpy()
cmaps_vec=['gray','gray','gray']
C_num_to_show=3
label_vec=["x","y","z"]
for j in range(0,C_num_to_show):
    plt.figure(figsize=(width, height))
    S0_channel_j=S0_to_plt[j,:,:]
    plt.imshow(S0_channel_j, cmap=cmaps_vec[j], interpolation='nearest')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_S0_dir + f"/S0_channel{j}.svg", format="svg", bbox_inches='tight', pad_inches=0)
    plt.close()

    # colorbar
    fig, ax = plt.subplots()  # Adjust the aspect ratio
    norm = plt.Normalize(vmin=S0_channel_j.min(), vmax=S0_channel_j.max())
    cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmaps_vec[j]),  # Use the 'Reds' colormap
        cax=ax,
        orientation='vertical'
    )
    cb.set_label('Value', fontsize=12)  # Add a label if desired
    plt.savefig(out_S0_dir + f"/colorbar_S0_channel{j}.svg")
    plt.close()
# plt.figure(figsize=(width, height))
# plt.imshow(S0_x,cmap='gray', interpolation='nearest')
# # Remove x and y ticks
# plt.axis('off')  # Turn off the axes
# plt.tight_layout()
# plt.savefig(inCoefDir+"S0_x.svg",format="svg", bbox_inches='tight', pad_inches=0)
# plt.close()
###########################################################
###########################################################

###########################################################

###########################################################
#plot Phi0 (this is T0 in paper)
cmaps_vec=["Reds","Oranges","Greens","Blues","Purples"]
in_Phi0_file=inCoefDir+"/Phi0.pth"
out_Phi0_dir=outCoefDir+"/Phi0Pics/"
Path(out_Phi0_dir).mkdir(parents=True, exist_ok=True)
Phi0=torch.load(in_Phi0_file)


print(f"Phi0.shape={Phi0.shape}")

C_num_to_show=5

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


###########################################################
#plt F1
cmaps_vec=["Reds","Oranges","Greens","Blues","Purples"]
in_F1_file=inCoefDir+"/F1.pth"
out_F1_dir=outCoefDir+"/F1Pics/"
Path(out_F1_dir).mkdir(parents=True, exist_ok=True)
F1=torch.load(in_F1_file)

C_num_to_show=5
F1_to_plt=F1[0,0:C_num_to_show,:,:].detach().numpy()

for j in range(0,C_num_to_show):
    plt.figure(figsize=(width, height))
    F1_channel_j=F1_to_plt[j,:,:]
    plt.imshow(F1_channel_j,cmap=cmaps_vec[j],interpolation='nearest')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_F1_dir+f"/F1_channel{j}.svg",format="svg", bbox_inches='tight', pad_inches=0)
    plt.close()

    #colorbar
    fig, ax = plt.subplots()  # Adjust the aspect ratio
    norm = plt.Normalize(vmin=F1_channel_j.min(), vmax=F1_channel_j.max())
    cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmaps_vec[j]),  # Use the 'Reds' colormap
        cax=ax,
        orientation='vertical'
    )
    cb.set_label('Value', fontsize=12)  # Add a label if desired
    plt.savefig(out_F1_dir + f"/colorbar_F1channel{j}.svg")
    plt.close()

###########################################################
#plt conv_W0
cmaps_vec=["Reds","Oranges","Greens","Blues","Purples"]
in_conv_W0_file=inCoefDir+"/conv_W0.pth"
out_conv_W0_dir=outCoefDir+"/conv_W0Pics/"
Path(out_conv_W0_dir).mkdir(parents=True, exist_ok=True)
conv_W0=torch.load(in_conv_W0_file)
print(f"conv_W0[0].shape={conv_W0[0].shape}")
C_num_to_show=5
conv_W0_to_plt=(conv_W0[0][0,0:C_num_to_show,:,:]).detach().numpy()
for j in range(0,C_num_to_show):
    plt.figure(figsize=(width, height))
    conv_W0_channel_j=conv_W0_to_plt[j,:,:]
    plt.imshow(conv_W0_channel_j,cmap=cmaps_vec[j],interpolation='nearest')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_conv_W0_dir+f"/conv_W0_channel{j}.svg",format="svg", bbox_inches='tight', pad_inches=0)
    plt.close()

    #colorbar
    fig, ax = plt.subplots()  # Adjust the aspect ratio
    norm = plt.Normalize(vmin=conv_W0_channel_j.min(), vmax=conv_W0_channel_j.max())
    cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmaps_vec[j]),  # Use the 'Reds' colormap
        cax=ax,
        orientation='vertical'
    )
    cb.set_label('Value', fontsize=12)  # Add a label if desired
    plt.savefig(out_conv_W0_dir + f"/colorbar_conv_W0channel{j}.svg")
    plt.close()
###########################################################

#plt conv_W1
cmaps_vec=["Reds","Oranges","Greens","Blues","Purples"]
in_conv_W1_file=inCoefDir+"/conv_W1.pth"
out_conv_W1_dir=outCoefDir+"/conv_W1Pics/"
Path(out_conv_W1_dir).mkdir(parents=True, exist_ok=True)
conv_W1=torch.load(in_conv_W1_file)
print(f"conv_W1[0].shape={conv_W1[0].shape}")
C_num_to_show=5
conv_W1_to_plt=(conv_W1[0][0,0:C_num_to_show,:,:]).detach().numpy()
for j in range(0,C_num_to_show):
    plt.figure(figsize=(width, height))
    conv_W1_channel_j=conv_W1_to_plt[j,:,:]
    plt.imshow(conv_W1_channel_j,cmap=cmaps_vec[j],interpolation='nearest')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_conv_W1_dir+f"/conv_W1_channel{j}.svg",format="svg", bbox_inches='tight', pad_inches=0)
    plt.close()

    #colorbar
    fig, ax = plt.subplots()  # Adjust the aspect ratio
    norm = plt.Normalize(vmin=conv_W1_channel_j.min(), vmax=conv_W1_channel_j.max())
    cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmaps_vec[j]),  # Use the 'Reds' colormap
        cax=ax,
        orientation='vertical'
    )
    cb.set_label('Value', fontsize=12)  # Add a label if desired
    plt.savefig(out_conv_W1_dir + f"/colorbar_conv_W1channel{j}.svg")
    plt.close()






###########################################################
#plt F1
cmaps_vec=["Reds","Oranges","Greens","Blues","Purples"]

in_T1_file=inCoefDir+"/T1.pth"
out_T1_dir=outCoefDir+"/T1Pics/"
Path(out_T1_dir).mkdir(parents=True, exist_ok=True)
T1=torch.load(in_T1_file)

C_num_to_show=5
T1_to_plt=T1[0,0:C_num_to_show,:,:].detach().numpy()

for j in range(0,C_num_to_show):
    plt.figure(figsize=(width, height))
    T1_channel_j=T1_to_plt[j,:,:]
    plt.imshow(T1_channel_j,cmap=cmaps_vec[j],interpolation='nearest')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_T1_dir+f"/T1_channel{j}.svg",format="svg", bbox_inches='tight', pad_inches=0)
    plt.close()

    #colorbar
    fig, ax = plt.subplots()  # Adjust the aspect ratio
    norm = plt.Normalize(vmin=T1_channel_j.min(), vmax=T1_channel_j.max())
    cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmaps_vec[j]),  # Use the 'Reds' colormap
        cax=ax,
        orientation='vertical'
    )
    cb.set_label('Value', fontsize=12)  # Add a label if desired
    plt.savefig(out_T1_dir + f"/colorbar_T1channel{j}.svg")
    plt.close()
###########################################################

###########################################################
#plt S1
cmaps_vec=["Reds","Oranges","Greens","Blues","Purples"]

in_S1_file=inCoefDir+"/S1.pth"
out_S1_dir=outCoefDir+"/S1Pics/"
Path(out_S1_dir).mkdir(parents=True, exist_ok=True)
S1=torch.load(in_S1_file)

C_num_to_show=5
S1_to_plt=S1[0,0:C_num_to_show,:,:].detach().numpy()

for j in range(0,C_num_to_show):
    plt.figure(figsize=(width, height))
    S1_channel_j=S1_to_plt[j,:,:]
    plt.imshow(S1_channel_j,cmap=cmaps_vec[j],interpolation='nearest')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_S1_dir+f"/S1_channel{j}.svg",format="svg", bbox_inches='tight', pad_inches=0)
    plt.close()

    #colorbar
    fig, ax = plt.subplots()  # Adjust the aspect ratio
    norm = plt.Normalize(vmin=S1_channel_j.min(), vmax=S1_channel_j.max())
    cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmaps_vec[j]),  # Use the 'Reds' colormap
        cax=ax,
        orientation='vertical'
    )
    cb.set_label('Value', fontsize=12)  # Add a label if desired
    plt.savefig(out_S1_dir + f"/colorbar_S1channel{j}.svg")
    plt.close()
###########################################################

###########################################################
#plot F2
cmaps_vec=["Reds","Oranges","Greens","Blues","Purples"]

in_F2_file=inCoefDir+"/F2.pth"
out_F2_dir=outCoefDir+"/F2Pics/"
Path(out_F2_dir).mkdir(parents=True, exist_ok=True)
F2=torch.load(in_F2_file)

C_num_to_show=5
F2_to_plt=F2[0,0:C_num_to_show,:,:].detach().numpy()

for j in range(0,C_num_to_show):
    plt.figure(figsize=(width, height))
    F2_channel_j=F2_to_plt[j,:,:]
    plt.imshow(F2_channel_j,cmap=cmaps_vec[j],interpolation='nearest')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_F2_dir+f"/F2_channel{j}.svg",format="svg", bbox_inches='tight', pad_inches=0)
    plt.close()

    #colorbar
    fig, ax = plt.subplots()  # Adjust the aspect ratio
    norm = plt.Normalize(vmin=F2_channel_j.min(), vmax=F2_channel_j.max())
    cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmaps_vec[j]),  # Use the 'Reds' colormap
        cax=ax,
        orientation='vertical'
    )
    cb.set_label('Value', fontsize=12)  # Add a label if desired
    plt.savefig(out_F2_dir + f"/colorbar_F2channel{j}.svg")
    plt.close()
###########################################################

###########################################################
#plot T2
cmaps_vec=["Reds","Oranges","Greens","Blues","Purples"]

in_T2_file=inCoefDir+"/T2.pth"
out_T2_dir=outCoefDir+"/T2Pics/"
Path(out_T2_dir).mkdir(parents=True, exist_ok=True)
T2=torch.load(in_T2_file)

C_num_to_show=5
T2_to_plt=T2[0,0:C_num_to_show,:,:].detach().numpy()

for j in range(0,C_num_to_show):
    plt.figure(figsize=(width, height))
    T2_channel_j=T2_to_plt[j,:,:]
    plt.imshow(T2_channel_j,cmap=cmaps_vec[j],interpolation='nearest')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_T2_dir+f"/T2_channel{j}.svg",format="svg", bbox_inches='tight', pad_inches=0)
    plt.close()

    #colorbar
    fig, ax = plt.subplots()  # Adjust the aspect ratio
    norm = plt.Normalize(vmin=T2_channel_j.min(), vmax=T2_channel_j.max())
    cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmaps_vec[j]),  # Use the 'Reds' colormap
        cax=ax,
        orientation='vertical'
    )
    cb.set_label('Value', fontsize=12)  # Add a label if desired
    plt.savefig(out_T2_dir + f"/colorbar_T2channel{j}.svg")
    plt.close()
###########################################################



###########################################################
#plot S2
cmaps_vec=["Reds","Oranges","Greens","Blues","Purples"]

in_S2_file=inCoefDir+"/S2.pth"
out_S2_dir=outCoefDir+"/S2Pics/"
Path(out_S2_dir).mkdir(parents=True, exist_ok=True)
S2=torch.load(in_S2_file)

C_num_to_show=5
S2_to_plt=S2[0,0:C_num_to_show,:,:].detach().numpy()

for j in range(0,C_num_to_show):
    plt.figure(figsize=(width, height))
    S2_channel_j=S2_to_plt[j,:,:]
    plt.imshow(S2_channel_j,cmap=cmaps_vec[j],interpolation='nearest')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_S2_dir+f"/S2_channel{j}.svg",format="svg", bbox_inches='tight', pad_inches=0)
    plt.close()

    #colorbar
    fig, ax = plt.subplots()  # Adjust the aspect ratio
    norm = plt.Normalize(vmin=S2_channel_j.min(), vmax=S2_channel_j.max())
    cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmaps_vec[j]),  # Use the 'Reds' colormap
        cax=ax,
        orientation='vertical'
    )
    cb.set_label('Value', fontsize=12)  # Add a label if desired
    plt.savefig(out_S2_dir + f"/colorbar_S2channel{j}.svg")
    plt.close()

###########################################################



###########################################################


###########################################################







