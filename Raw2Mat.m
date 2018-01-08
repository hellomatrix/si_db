clear all;clc;close all;
%%
[Readfilename, Readpathname] = uigetfile('*.raw', 'choose a file','*.raw','MultiSelect','off');
cd(Readpathname);
FileName = [Readpathname Readfilename];
fid = fopen(FileName,'rb');
%%
%%
frame=1;
while ~feof(fid)
status = feof(fid);
if status==1
    break;
end
[DataOrigin,count]=fread(fid,[5534720,1],'uint16');
img1=zeros(2162,2560);
t=1;
for i=1:2162
    for j=1:2560
        img1(i,j)=DataOrigin(t);
        t=t+1;
    end
end
imshow(img1,[]);
file=['ตฺ',num2str(frame),'ึก.mat'];
 save(file,'img1');
frame=frame+1;
end
fclose(fid);


