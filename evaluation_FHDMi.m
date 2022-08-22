clear all;
clc ;

result_dir_name = 'result/FHDMi/DMSFN_plus' ; 
out_excel_name = 'FHDMi_DMSFN_plus' ; 
title_Isrc = sprintf('%s%s%s', './',result_dir_name , '/output/') ;
title_Iout = sprintf('%s%s', './' , 'datasets/FHDMi/test/target/') ; 

addpath(title_Isrc); 
addpath(title_Iout);

dpath_input = [dir([title_Isrc '*' 'bmp']) ; dir([title_Isrc '*' 'jpg']) ; dir([title_Isrc '*' 'png'])]; 
dpath_gt = [dir([title_Iout '*' 'bmp']) ; dir([title_Iout '*' 'jpg']) ; dir([title_Iout '*' 'png']) ]; 


xlsFile = sprintf('%s%s',out_excel_name , '.xls') ;

sheetName = "psnr" ; 
data={'PIC','PSNR','SSIM'};

ssim_demoire = 0;
psnr_demoire = 0;
res_ssim = 0;
res_psnr = 0;
k = 1  ;

for i = 1 : 1 : length(dpath_input)
    
     img_input = sprintf('%s%s', title_Isrc, dpath_input(i).name); 
     img_gt = sprintf('%s%s', title_Iout, dpath_gt(i).name); 
   
     im_source=im2double(imread(img_input));   
     im_target=im2double(imread(img_gt));  
     

    img_ssim = ssim(im_source,im_target);
    img_psnr = psnr(im_source,im_target);

    ssim_demoire = ssim_demoire + img_ssim ; 
    psnr_demoire = psnr_demoire + img_psnr ; 
 

    data{i+2,1}= dpath_input(i).name;
    data{i+2,2}=img_psnr;
    data{i+2,3}=img_ssim;
    

    res_ssim = ssim_demoire / k;
    res_psnr = psnr_demoire / k;
    k = i+1;
    
    disp(k)

end

[status, message] = xlswrite(xlsFile, data, sheetName);
dos(['start ' xlsFile]);

sprintf('%s%3.3f', "Average SSIM :", res_ssim)
sprintf('%s%3.3f', "Average PSNR :", res_psnr)

    
    