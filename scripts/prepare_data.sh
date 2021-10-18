#!/usr/bin/env bash

###################### Collecting split AMR data_utils into train dev and test respectively.  ###################

# AMR 1.0 and 2.0 have different path structures.
# Collecting them in different ways.

set -e

usage() {
    echo "Usage: $0 -v <AMR corpus version:1 or 2> -p <Path to AMR corpus>"
    exit 1;
}

while getopts ":h:v:p:" o; do
    case "${o}" in
        h)
            usage
            ;;
        v)
            v=${OPTARG}
            ((v == 1 || v == 2)) || usage
            ;;
        p)
            p=${OPTARG}
            ;;
        \? )
            usage
            ;;
    esac
done
shift $((OPTIND-1))

if [ -z $v ]; then
    usage
fi

if [ -z $p ]; then
    usage
fi


if [[ "$v" == "2" ]]; then
    DATA_DIR=data/AMR/amr_2.0
    SPLIT_DIR=$p/data/amrs/split
    TRAIN=${SPLIT_DIR}/training
    DEV=${SPLIT_DIR}/dev
    TEST=${SPLIT_DIR}/test
else
    DATA_DIR=data/AMR/amr_1.0
    SPLIT_DIR=$p/data/split
    TRAIN=${SPLIT_DIR}/training
    DEV=${SPLIT_DIR}/dev
    TEST=${SPLIT_DIR}/test
fi

echo -e "\e[5;92mPreparing data_utils in ${DATA_DIR}...\e[0m"
mkdir -p ${DATA_DIR}
awk FNR!=1 ${TRAIN}/* > ${DATA_DIR}/train.txt
awk FNR!=1 ${DEV}/* > ${DATA_DIR}/dev.txt
awk FNR!=1 ${TEST}/* > ${DATA_DIR}/test.txt


clear



####################### Downloading artifacts. ###################
# Download bert-base pre-trained embeddings
echo -e "\e[5;92mDownloading Bert Pre-trained Embeddings...\e[0m"
mkdir -p data/bert-base-cased
curl --progress-bar -O https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz
tar -xzvf bert-base-cased.tar.gz -C data/bert-base-cased
curl --progress-bar -o data/bert-base-cased/bert-base-cased-vocab.txt \
    https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-vocab.txt
rm bert-base-cased.tar.gz

clear


# Download Glove embeddings.
echo -e "\e[5;92mDownloading Glove...\e[0m"

mkdir -p data/glove
curl --progress-bar -L -o  data/glove/glove.840B.300d.zip \
        http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip

clear

echo -e "\e[5;92mDownloading evaluation tools...\e[0m"
# Get amr evaluation tools witch needs python 2
mkdir -p evaluation_tools
git clone https://github.com/ChunchuanLv/amr-evaluation-tool-enhanced.git evaluation_tools/amr-evaluation-tool-enhanced


clear


echo -e "\e[5;92mDownloading amr utils (From STOG's open source)...\e[0m"
# These utils contains NE information and other useful data_utils for AMR re-construction
mkdir -p data/AMR
curl --progress-bar -o  data/AMR/amr_2.0_utils.tar.gz https://www.cs.jhu.edu/~s.zhang/data/AMR/amr_2.0_utils.tar.gz
curl --progress-bar -o  data/AMR/amr_1.0_utils.tar.gz https://www.cs.jhu.edu/~s.zhang/data/AMR/amr_1.0_utils.tar.gz
pushd data/AMR
tar -xzvf amr_2.0_utils.tar.gz
tar -xzvf amr_1.0_utils.tar.gz
rm amr_2.0_utils.tar.gz amr_1.0_utils.tar.gz
popd



################################################

echo -e "\e[1;92mFinishing Data Preparing! \n`date` \e[0m"
