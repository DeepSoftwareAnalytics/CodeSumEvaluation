#!/usr/bin/env bash

echo "Downloading TL-CodeSum dataset"
FILE=master.zip
if [[ -f "$FILE" ]]; then
    echo "$FILE exists, skipping download"
else
    wget https://github.com/xing-hu/TL-CodeSum/archive/master.zip
    unzip ${FILE} && rm ${FILE}
fi

mv TL-CodeSum-master/data/ original
rm -rf TL-CodeSum-master
