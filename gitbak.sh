#!/bin/bash

path="$(cd $(dirname $0);pwd)"

function upload() {
    now_time=$(date '+%Y-%m-%d %H:%M:%S')
    echo $path
    cd $path
    echo "[info] entering git path... $(pwd)"
    git add -A
    git commit -m "Time: $now_time"
    git push origin
}


function main() {
    upload
}

main