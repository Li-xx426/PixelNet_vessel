clear all
clc
imagelist=textread('./image_zpd.txt','%s');
seglist=textread('./label_zpd.txt','%s');
save drive_zpd imagelist seglist
