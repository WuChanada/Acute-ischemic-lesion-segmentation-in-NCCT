close all

%clc
addpath('NIfTI_20140122');


feat = [];
trlabel = [];



rootdir = '/media/wu/Data/Acute_kss_data/LeftLesion/';%img and feat path
p = dir([rootdir '0*']);
for i=1:size(p,1)
    
    name = p(i).name;
   workingDIR = [rootdir name '/'];

    
   niiPath = ['LeftSamples/' p(i).name 'v2.nii.gz'];%groundtruth
    niiImage = load_untouch_nii(niiPath);
    gt = double(niiImage.img);
     niiPath = [workingDIR 'NCCT_brain_mni.nii.gz'];% original NCCT in MNI space
    niiImage = load_untouch_nii(niiPath);
    maskim = double(niiImage.img);
    
    niiSaveName = [workingDIR 'NCCT_brain_mni.nii.gz'];% original NCCT in MNI space
    nii=load_untouch_nii(niiSaveName);
  im=double(nii.img);
  
  % img med 3x3x3
  im3 = medfilt3(im,[3 3 3]);
  ind=find(maskim>10&maskim<50);
  ave0=mean(im3(ind));
  std0=std(im3(ind));
  im3=(im3-ave0)/(std0+eps);
   im3 = (im3-min(im3(:)))./(max(im3(:))-min(im3(:)));
   
  % img med 7x7x7
  im7 = medfilt3(im,[7 7 7]);
    ind=find(maskim>10&maskim<50);
  ave0=mean(im7(ind));
  std0=std(im7(ind));
  im7=(im7-ave0)/(std0+eps);
   im7 = (im7-min(im7(:)))./(max(im7(:))-min(im7(:)));
  % img med 11x11x11
  im11 = medfilt3(im,[11 11 11]);
  ind=find(maskim>10&maskim<50);
  ave0=mean(im11(ind));
  std0=std(im11(ind));
  im11=(im11-ave0)/(std0+eps);
   im11 = (im11-min(im11(:)))./(max(im11(:))-min(im11(:)));
  


   

 niiPath = [workingDIR 'Diff_flipx_reg_Be0.005.nii.gz'];%difference map
    niiImage = load_untouch_nii(niiPath);
    dif = double(niiImage.img);
    hdr = niiImage.hdr;
    dif0=dif;
  
      % difference med 3x3x3
 dif3 = medfilt3(dif,[3 3 3]);
    dif3(dif3>=5)=5;
    dif3(dif3<=-5)=-5;
          dif3 = (dif3-min(dif3(:)))./(max(dif3(:))-min(dif3(:)));
  % difference med 7x7x7
  dif7 = medfilt3(dif,[7 7 7]);
   dif7(dif7>=5)=5;
    dif7(dif7<=-5)=-5;
          dif7 = (dif7-min(dif7(:)))./(max(dif7(:))-min(dif7(:)));
  % difference med 11x11x11
  dif11 = medfilt3(dif,[11 11 11]);
   dif11(dif11>=5)=5;
    dif11(dif11<=-5)=-5;
          dif11 = (dif11-min(dif11(:)))./(max(dif11(:))-min(dif11(:)));
    
 
         

   niiPath = [workingDIR '_locatProbR.nii.gz'];%location prob
    niiImage = load_untouch_nii(niiPath);
    hdr = niiImage.hdr;
     locfeat = double(niiImage.img);

    
      niiSaveName = [workingDIR 'discost.nii.gz'];% distance from CSF
    nii=load_untouch_nii(niiSaveName);
  distfeat=double(nii.img);
 negidx0= find((maskim>10 & maskim<50)&gt==2);

  posidx0= find((maskim>10 & maskim<50)&gt==1&dif0<=-1.5);
  idx = [negidx0;posidx0];

trlabel=[trlabel; gt(idx)];
trlabel(trlabel==2) = 0;
% tmp = [imfeat(idx) imdiffeat(idx) locfeat(idx) distfeat(idx)]; %locfeat(idx)  eqfeat(idx) eqdiffeat(idx)  distfeat(idx)
tmp = [im3(idx) im7(idx) im11(idx) dif3(idx) dif7(idx) dif11(idx) distfeat(idx) locfeat(idx)];
feat = [feat;tmp];


disp(i)

end
feat(isnan(feat))=0;
save('train_feat.mat','feat','trlabel')
