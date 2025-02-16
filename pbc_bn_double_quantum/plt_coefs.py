import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
inCoefDir="./coefs/"
#this script plots layers

in_S0_file=inCoefDir+"/S0.pth"
width=4
height=4
S0=torch.load(in_S0_file)
# print(S0.shape)

S0=S0.detach().cpu().numpy()
S0_x=np.array(S0[0,0,:,:])
S0_y=np.array(S0[0,1,:,:])
S0_z=np.array(S0[0,2,:,:])
out_S0_dir=inCoefDir+"/S0Pics/"
Path(out_S0_dir).mkdir(parents=True, exist_ok=True)
cmaps_vec=['YlOrRd','YlGn','PuBu']# for S0
###########################################################
#plot S0_x
plt.figure(figsize=(width, height))
plt.imshow(S0_x,cmap=cmaps_vec[0], interpolation='nearest')
# Remove x and y ticks
plt.axis('off')  # Turn off the axes
plt.tight_layout()
plt.savefig(out_S0_dir+"S0_x.svg",format="svg", bbox_inches='tight', pad_inches=0)
plt.close()
#colorbar S0_x
fig, ax = plt.subplots()  # Adjust the aspect ratio
norm = plt.Normalize(vmin=S0_x.min(), vmax=S0_x.max())
cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmaps_vec[0]),  # Use the 'Reds' colormap
        cax=ax,
        orientation='vertical'
    )
cb.set_label('Value', fontsize=12)  # Add a label if desired
plt.savefig(out_S0_dir+f"/colorbar_x.svg")
plt.close()


#plot S0_y
plt.figure(figsize=(width, height))
plt.imshow(S0_y,cmap=cmaps_vec[1], interpolation='nearest')
# Remove x and y ticks
plt.axis('off')  # Turn off the axes
plt.tight_layout()
plt.savefig(out_S0_dir+"S0_y.svg",format="svg", bbox_inches='tight', pad_inches=0)
plt.close()
#colorbar S0_y
fig, ax = plt.subplots()  # Adjust the aspect ratio
norm = plt.Normalize(vmin=S0_y.min(), vmax=S0_y.max())
cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmaps_vec[1]),  # Use the 'Reds' colormap
        cax=ax,
        orientation='vertical'
    )
cb.set_label('Value', fontsize=12)  # Add a label if desired
plt.savefig(out_S0_dir+f"/colorbar_y.svg")
plt.close()


#plot S0_z
plt.figure(figsize=(width, height))
plt.imshow(S0_z,cmap=cmaps_vec[2], interpolation='nearest')
# Remove x and y ticks
plt.axis('off')  # Turn off the axes
plt.tight_layout()
plt.savefig(out_S0_dir+"S0_z.svg",format="svg", bbox_inches='tight', pad_inches=0)
plt.close()
#colorbar S0_z
fig, ax = plt.subplots()  # Adjust the aspect ratio
norm = plt.Normalize(vmin=S0_z.min(), vmax=S0_z.max())
cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmaps_vec[2]),  # Use the 'Reds' colormap
        cax=ax,
        orientation='vertical'
    )
cb.set_label('Value', fontsize=12)  # Add a label if desired
plt.savefig(out_S0_dir+f"/colorbar_z.svg")
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
Phi0_to_plt=Phi0[0,0:C_num_to_show,:,:].detach().cpu().numpy()

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


###########################################################

###########################################################
#plot g1

cmaps_vec=["Reds","Oranges","Greens","Blues","Purples"]
in_g1_file=inCoefDir+"/g1.pth"
out_g1_dir=inCoefDir+"/g1Pics/"
Path(out_g1_dir).mkdir(parents=True, exist_ok=True)
g1=torch.load(in_g1_file)
C_num_to_show=5
print(f"g1.shape={g1.shape}")
g1_to_plt=g1[0,0:C_num_to_show,:,:].detach().cpu().numpy()

##plot g1, channel0
# g1_channel0=g1_to_plt[0,:,:]
# plt.figure(figsize=(width, height))
# plt.imshow(g1_channel0, cmap=cmaps_vec[0], interpolation='nearest')  # Use the Reds colormap
# plt.axis('off')  # Turn off the axes
# plt.tight_layout()
# plt.savefig(out_g1_dir+"/g1_channel0.svg",format="svg", bbox_inches='tight', pad_inches=0)
# plt.close()
for j in range(0,C_num_to_show):
    plt.figure(figsize=(width, height))
    g1_channel_j=g1_to_plt[j,:,:]
    plt.imshow(g1_channel_j,cmap=cmaps_vec[j],interpolation='nearest')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_g1_dir + f"/g1_channel{j}.svg", format="svg", bbox_inches='tight', pad_inches=0)
    plt.close()

    #colorbar
    fig, ax = plt.subplots()  # Adjust the aspect ratio
    norm = plt.Normalize(vmin=g1_channel_j.min(), vmax=g1_channel_j.max())
    cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmaps_vec[j]),  # Use the 'Reds' colormap
        cax=ax,
        orientation='vertical'
    )
    cb.set_label('Value', fontsize=12)  # Add a label if desired
    plt.savefig(out_g1_dir+f"/colorbar_g1_channel{j}.svg")
    plt.close()
###########################################################
#plot F1
cmaps_vec=["Reds","Oranges","Greens","Blues","Purples"]
in_F1_file=inCoefDir+"/F1.pth"
out_F1_dir=inCoefDir+"/F1Pics/"
Path(out_F1_dir).mkdir(parents=True, exist_ok=True)
F1=torch.load(in_F1_file)
C_num_to_show=5
print(f"F1.shape={F1.shape}")
F1_to_plt=F1[0,0:C_num_to_show,:,:].detach().cpu().numpy()

##plot F1, channel0
# F1_channel0=F1_to_plt[0,:,:]
# plt.figure(figsize=(width, height))
# plt.imshow(F1_channel0, cmap=cmaps_vec[0], interpolation='nearest')  # Use the Reds colormap
# plt.axis('off')  # Turn off the axes
# plt.tight_layout()
# plt.savefig(out_F1_dir+"/F1_channel0.svg",format="svg", bbox_inches='tight', pad_inches=0)
# plt.close()
for j in range(0,C_num_to_show):
    plt.figure(figsize=(width, height))
    F1_channel_j=F1_to_plt[j,:,:]
    plt.imshow(F1_channel_j,cmap=cmaps_vec[j],interpolation='nearest')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_F1_dir + f"/F1_channel{j}.svg", format="svg", bbox_inches='tight', pad_inches=0)
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
    plt.savefig(out_F1_dir+f"/colorbar_F1_channel{j}.svg")
    plt.close()

###########################################################
#plot T1
in_T1_file=inCoefDir+"/T1.pth"
out_T1_dir=inCoefDir+"/T1Pics/"
Path(out_T1_dir).mkdir(parents=True, exist_ok=True)
T1=torch.load(in_T1_file)
C_num_to_show=5
print(f"T1.shape={T1.shape}")
T1_to_plt=T1[0,0:C_num_to_show,:,:].detach().cpu().numpy()

##plot T1, channel0
# T1_channel0=T1_to_plt[0,:,:]
# plt.figure(figsize=(width, height))
# plt.imshow(T1_channel0, cmap=cmaps_vec[0], interpolation='nearest')  # Use the Reds colormap
# plt.axis('off')  # Turn off the axes
# plt.tight_layout()
# plt.savefig(out_T1_dir+"/T1_channel0.svg",format="svg", bbox_inches='tight', pad_inches=0)
# plt.close()
for j in range(0,C_num_to_show):
    plt.figure(figsize=(width, height))
    T1_channel_j=T1_to_plt[j,:,:]
    plt.imshow(T1_channel_j,cmap=cmaps_vec[j],interpolation='nearest')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_T1_dir + f"/T1_channel{j}.svg", format="svg", bbox_inches='tight', pad_inches=0)
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
    plt.savefig(out_T1_dir+f"/colorbar_T1_channel{j}.svg")
    plt.close()

###########################################################
#plot S1
in_S1_file=inCoefDir+"/S1.pth"
out_S1_dir=inCoefDir+"/S1Pics/"
Path(out_S1_dir).mkdir(parents=True, exist_ok=True)
S1=torch.load(in_S1_file)
C_num_to_show=5
print(f"S1.shape={S1.shape}")
S1_to_plt=S1[0,0:C_num_to_show,:,:].detach().cpu().numpy()

##plot S1, channel0
# S1_channel0=S1_to_plt[0,:,:]
# plt.figure(figsize=(width, height))
# plt.imshow(S1_channel0, cmap=cmaps_vec[0], interpolation='nearest')  # Use the Reds colormap
# plt.axis('off')  # Turn off the axes
# plt.tight_layout()
# plt.savefig(out_S1_dir+"/S1_channel0.svg",format="svg", bbox_inches='tight', pad_inches=0)
# plt.close()
for j in range(0,C_num_to_show):
    plt.figure(figsize=(width, height))
    S1_channel_j=S1_to_plt[j,:,:]
    plt.imshow(S1_channel_j,cmap=cmaps_vec[j],interpolation='nearest')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_S1_dir + f"/S1_channel{j}.svg", format="svg", bbox_inches='tight', pad_inches=0)
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
    plt.savefig(out_S1_dir+f"/colorbar_S1_channel{j}.svg")
    plt.close()


###########################################################
#plot F2
in_F2_file=inCoefDir+"/F2.pth"
out_F2_dir=inCoefDir+"/F2Pics/"
Path(out_F2_dir).mkdir(parents=True, exist_ok=True)
F2=torch.load(in_F2_file)
C_num_to_show=5
print(f"F2.shape={F2.shape}")
F2_to_plt=F2[0,0:C_num_to_show,:,:].detach().cpu().numpy()

##plot F2, channel0
# F2_channel0=F2_to_plt[0,:,:]
# plt.figure(figsize=(width, height))
# plt.imshow(F2_channel0, cmap=cmaps_vec[0], interpolation='nearest')  # Use the Reds colormap
# plt.axis('off')  # Turn off the axes
# plt.tight_layout()
# plt.savefig(out_F2_dir+"/F2_channel0.svg",format="svg", bbox_inches='tight', pad_inches=0)
# plt.close()
for j in range(0,C_num_to_show):
    plt.figure(figsize=(width, height))
    F2_channel_j=F2_to_plt[j,:,:]
    plt.imshow(F2_channel_j,cmap=cmaps_vec[j],interpolation='nearest')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_F2_dir + f"/F2_channel{j}.svg", format="svg", bbox_inches='tight', pad_inches=0)
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
    plt.savefig(out_F2_dir+f"/colorbar_F2_channel{j}.svg")
    plt.close()


###########################################################
#plot T2
in_T2_file=inCoefDir+"/T2.pth"
out_T2_dir=inCoefDir+"/T2Pics/"
Path(out_T2_dir).mkdir(parents=True, exist_ok=True)
T2=torch.load(in_T2_file)
C_num_to_show=5
print(f"T2.shape={T2.shape}")
T2_to_plt=T2[0,0:C_num_to_show,:,:].detach().cpu().numpy()

##plot T2, channel0
# T2_channel0=T2_to_plt[0,:,:]
# plt.figure(figsize=(width, height))
# plt.imshow(T2_channel0, cmap=cmaps_vec[0], interpolation='nearest')  # Use the Reds colormap
# plt.axis('off')  # Turn off the axes
# plt.tight_layout()
# plt.savefig(out_T2_dir+"/T2_channel0.svg",format="svg", bbox_inches='tight', pad_inches=0)
# plt.close()
for j in range(0,C_num_to_show):
    plt.figure(figsize=(width, height))
    T2_channel_j=T2_to_plt[j,:,:]
    plt.imshow(T2_channel_j,cmap=cmaps_vec[j],interpolation='nearest')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_T2_dir + f"/T2_channel{j}.svg", format="svg", bbox_inches='tight', pad_inches=0)
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
    plt.savefig(out_T2_dir+f"/colorbar_T2_channel{j}.svg")
    plt.close()


###########################################################
#plot S2
in_S2_file=inCoefDir+"/S2.pth"
out_S2_dir=inCoefDir+"/S2Pics/"
Path(out_S2_dir).mkdir(parents=True, exist_ok=True)
S2=torch.load(in_S2_file)
C_num_to_show=5
print(f"S2.shape={S2.shape}")
S2_to_plt=S2[0,0:C_num_to_show,:,:].detach().cpu().numpy()

##plot S2, channel0
# S2_channel0=S2_to_plt[0,:,:]
# plt.figure(figsize=(width, height))
# plt.imshow(S2_channel0, cmap=cmaps_vec[0], interpolation='nearest')  # Use the Reds colormap
# plt.axis('off')  # Turn off the axes
# plt.tight_layout()
# plt.savefig(out_S2_dir+"/S2_channel0.svg",format="svg", bbox_inches='tight', pad_inches=0)
# plt.close()
for j in range(0,C_num_to_show):
    plt.figure(figsize=(width, height))
    S2_channel_j=S2_to_plt[j,:,:]
    plt.imshow(S2_channel_j,cmap=cmaps_vec[j],interpolation='nearest')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_S2_dir + f"/S2_channel{j}.svg", format="svg", bbox_inches='tight', pad_inches=0)
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
    plt.savefig(out_S2_dir+f"/colorbar_S2_channel{j}.svg")
    plt.close()


###########################################################
#plot F3
# in_F3_file=inCoefDir+"/F3.pth"
# out_F3_dir=inCoefDir+"/F3Pics/"
# Path(out_F3_dir).mkdir(parents=True, exist_ok=True)
# F3=torch.load(in_F3_file)
# C_num_to_show=5
# print(f"F3.shape={F3.shape}")
# F3_to_plt=F3[0,0:C_num_to_show,:,:].detach().cpu().numpy()
#
# ##plot F3, channel0
# # F3_channel0=F3_to_plt[0,:,:]
# # plt.figure(figsize=(width, height))
# # plt.imshow(F3_channel0, cmap=cmaps_vec[0], interpolation='nearest')  # Use the Reds colormap
# # plt.axis('off')  # Turn off the axes
# # plt.tight_layout()
# # plt.savefig(out_F3_dir+"/F3_channel0.svg",format="svg", bbox_inches='tight', pad_inches=0)
# # plt.close()
# for j in range(0,C_num_to_show):
#     plt.figure(figsize=(width, height))
#     F3_channel_j=F3_to_plt[j,:,:]
#     plt.imshow(F3_channel_j,cmap=cmaps_vec[j],interpolation='nearest')
#     plt.axis('off')
#     plt.tight_layout()
#     plt.savefig(out_F3_dir + f"/F3_channel{j}.svg", format="svg", bbox_inches='tight', pad_inches=0)
#     plt.close()
#
#     #colorbar
#     fig, ax = plt.subplots()  # Adjust the aspect ratio
#     norm = plt.Normalize(vmin=F3_channel_j.min(), vmax=F3_channel_j.max())
#     cb = plt.colorbar(
#         plt.cm.ScalarMappable(norm=norm, cmap=cmaps_vec[j]),  # Use the 'Reds' colormap
#         cax=ax,
#         orientation='vertical'
#     )
#     cb.set_label('Value', fontsize=12)  # Add a label if desired
#     plt.savefig(out_F3_dir+f"/colorbar_F3_channel{j}.svg")
#     plt.close()


###########################################################
#plot T3
# in_T3_file=inCoefDir+"/T3.pth"
# out_T3_dir=inCoefDir+"/T3Pics/"
# Path(out_T3_dir).mkdir(parents=True, exist_ok=True)
# T3=torch.load(in_T3_file)
# C_num_to_show=5
# print(f"T3.shape={T3.shape}")
# T3_to_plt=T3[0,0:C_num_to_show,:,:].detach().cpu().numpy()
#
# ##plot T3, channel0
# # T3_channel0=T3_to_plt[0,:,:]
# # plt.figure(figsize=(width, height))
# # plt.imshow(T3_channel0, cmap=cmaps_vec[0], interpolation='nearest')  # Use the Reds colormap
# # plt.axis('off')  # Turn off the axes
# # plt.tight_layout()
# # plt.savefig(out_T3_dir+"/T3_channel0.svg",format="svg", bbox_inches='tight', pad_inches=0)
# # plt.close()
# for j in range(0,C_num_to_show):
#     plt.figure(figsize=(width, height))
#     T3_channel_j=T3_to_plt[j,:,:]
#     plt.imshow(T3_channel_j,cmap=cmaps_vec[j],interpolation='nearest')
#     plt.axis('off')
#     plt.tight_layout()
#     plt.savefig(out_T3_dir + f"/T3_channel{j}.svg", format="svg", bbox_inches='tight', pad_inches=0)
#     plt.close()
#
#     #colorbar
#     fig, ax = plt.subplots()  # Adjust the aspect ratio
#     norm = plt.Normalize(vmin=T3_channel_j.min(), vmax=T3_channel_j.max())
#     cb = plt.colorbar(
#         plt.cm.ScalarMappable(norm=norm, cmap=cmaps_vec[j]),  # Use the 'Reds' colormap
#         cax=ax,
#         orientation='vertical'
#     )
#     cb.set_label('Value', fontsize=12)  # Add a label if desired
#     plt.savefig(out_T3_dir+f"/colorbar_T3_channel{j}.svg")
#     plt.close()

###########################################################
#plot S3
# in_S3_file=inCoefDir+"/S3.pth"
# out_S3_dir=inCoefDir+"/S3Pics/"
# Path(out_S3_dir).mkdir(parents=True, exist_ok=True)
# S3=torch.load(in_S3_file)
# C_num_to_show=5
# print(f"S3.shape={S3.shape}")
# S3_to_plt=S3[0,0:C_num_to_show,:,:].detach().cpu().numpy()
#
# ##plot S3, channel0
# # S3_channel0=S3_to_plt[0,:,:]
# # plt.figure(figsize=(width, height))
# # plt.imshow(S3_channel0, cmap=cmaps_vec[0], interpolation='nearest')  # Use the Reds colormap
# # plt.axis('off')  # Turn off the axes
# # plt.tight_layout()
# # plt.savefig(out_S3_dir+"/S3_channel0.svg",format="svg", bbox_inches='tight', pad_inches=0)
# # plt.close()
# for j in range(0,C_num_to_show):
#     plt.figure(figsize=(width, height))
#     S3_channel_j=S3_to_plt[j,:,:]
#     plt.imshow(S3_channel_j,cmap=cmaps_vec[j],interpolation='nearest')
#     plt.axis('off')
#     plt.tight_layout()
#     plt.savefig(out_S3_dir + f"/S3_channel{j}.svg", format="svg", bbox_inches='tight', pad_inches=0)
#     plt.close()
#
#     #colorbar
#     fig, ax = plt.subplots()  # Adjust the aspect ratio
#     norm = plt.Normalize(vmin=S3_channel_j.min(), vmax=S3_channel_j.max())
#     cb = plt.colorbar(
#         plt.cm.ScalarMappable(norm=norm, cmap=cmaps_vec[j]),  # Use the 'Reds' colormap
#         cax=ax,
#         orientation='vertical'
#     )
#     cb.set_label('Value', fontsize=12)  # Add a label if desired
#     plt.savefig(out_S3_dir+f"/colorbar_S3_channel{j}.svg")
#     plt.close()


###########################################################
#plot final output
final_output_cmap="gray"
in_final_output_file=inCoefDir+"/final_output.pth"
out_final_output_dir=inCoefDir+"/final_outputPics/"
Path(out_final_output_dir).mkdir(parents=True, exist_ok=True)
final_output=torch.load(in_final_output_file)
plt_final_output=final_output[0,:,:].detach().cpu().numpy()
print(f"plt_final_output.shape={plt_final_output.shape}")

#plt final output
plt.figure(figsize=(width, height))
plt.imshow(plt_final_output,cmap=final_output_cmap, interpolation='nearest')
# Remove x and y ticks
plt.axis('off')  # Turn off the axes
plt.tight_layout()
plt.savefig(out_final_output_dir+"final_output.svg",format="svg", bbox_inches='tight', pad_inches=0)
plt.close()

#colorbar plt_final_output
fig, ax = plt.subplots()  # Adjust the aspect ratio
norm = plt.Normalize(vmin=plt_final_output.min(), vmax=plt_final_output.max())
cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=final_output_cmap),  # Use the 'Reds' colormap
        cax=ax,
        orientation='vertical'
    )
cb.set_label('Value', fontsize=12)  # Add a label if desired
plt.savefig(out_final_output_dir+f"/colorbar_final_output.svg")
plt.close()