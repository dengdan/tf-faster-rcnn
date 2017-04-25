#!/bin/bash

DIR=~/models/faster-rcnn
cd $DIR

FILE=coco_long.tgz
URL=http://ladoga.graphics.cs.cmu.edu/xinleic/tf-faster-rcnn/$FILE
CHECKSUM=099f637235d24ec97d9708b3ff66bd7f

if [ -f $FILE ]; then
  echo "File already exists. Checking md5..."
  os=`uname -s`
  if [ "$os" = "Linux" ]; then
    checksum=`md5sum $FILE | awk '{ print $1 }'`
  elif [ "$os" = "Darwin" ]; then
    checksum=`cat $FILE | md5`
  fi
  if [ "$checksum" = "$CHECKSUM" ]; then
    echo "Checksum is correct. No need to download."
    exit 0
  else
    echo "Checksum is incorrect. Need to download again."
  fi
fi

echo "Downloading Faster R-CNN models for COCO with longer training (1G)..."

proxychains wget $URL -O $FILE

echo "Unzipping..."

tar zxvf $FILE

echo "Done. Please run this command again to verify that checksum = $CHECKSUM."