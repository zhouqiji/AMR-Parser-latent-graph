#!/usr/bin/env bash


set -e

usage() {
    echo "Usage: $0 -v <AMR corpus version:1 or 2>  -p <generated AMR file>"
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
    # Directory where intermediate utils will be saved to speed up processing.
    util_dir=data/AMR/amr_2.0_utils

    # AMR data with **features**
    data_dir=data/AMR/amr_2.0
    test_data=$p

    # ========== Set the above variables correctly ==========

    printf "Frame lookup...`date`\n"
    python -u -m zamr.data_utils.dataset_readers.amr_parsing.postprocess.node_restore \
        --amr_files ${test_data} \
        --util_dir ${util_dir}
    printf "Done.`date`\n\n"

    printf "Wikification...`date`\n"
    python -u -m zamr.data_utils.dataset_readers.amr_parsing.postprocess.wikification \
        --amr_files ${test_data}.frame \
        --util_dir ${util_dir}
    printf "Done.`date`\n\n"

    printf "Expanding nodes...`date`\n"
    python -u -m zamr.data_utils.dataset_readers.amr_parsing.postprocess.expander \
        --amr_files ${test_data}.frame.wiki \
        --util_dir ${util_dir}
    printf "Done.`date`\n\n"

    mv ${test_data}.frame.wiki.expand ${test_data}
    rm ${test_data}.frame*

else
    # Directory where intermediate utils will be saved to speed up processing.
    util_dir=data/AMR/amr_1.0_utils

    # AMR data with **features**
    test_data=$p

    # ========== Set the above variables correctly ==========

    printf "Frame lookup...`date`\n"
    python -u -m zamr.data_utils.dataset_readers.amr_parsing.postprocess.node_restore \
        --amr_files ${test_data} \
        --util_dir ${util_dir} || exit
    printf "Done.`date`\n\n"

    printf "Expanding nodes...`date`\n"
    python -u -m zamr.data_utils.dataset_readers.amr_parsing.postprocess.expander \
        --amr_files ${test_data}.frame \
        --util_dir ${util_dir} || exit
    printf "Done.`date`\n\n"

mv ${test_data}.frame.expand ${test_data}
rm ${test_data}.frame*



fi
