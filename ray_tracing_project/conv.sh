# usage: bash converter.sh [path-to-data-images] [path-to-save-jpgs]

nvcc main.cu
./a.out --cpu
mkdir -p $2
# rm $2/*
for file in $1/*
do 
    f=${file#$1/};
    echo Converting $file to $2/${f%.data}.jpg
    python3 conv.py $file $2/${f%.data}.jpg 
done