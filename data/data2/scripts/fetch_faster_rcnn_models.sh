#!/bin/bash

DIR=~/models/faster-rcnn
cd $DIR

FILE=faster_rcnn_models.tgz
URL=http://ladoga.graphics.cs.cmu.edu/xinleic/tf-faster-rcnn/$FILE
CHECKSUM=865cdf7350a87ef41d6476e6e33b7212

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

echo "Downloading Faster R-CNN models (2G)..."

proxychains wget $URL -O $FILE

echo "Unzipping..."

tar zxvf $FILE

echo "Done. Please run this command again to verify that checksum = $CHECKSUM."
