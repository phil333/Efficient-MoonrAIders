#!/bin/bash

## Remove a link(s) from an RTAB Map database and re-evaluate it

PATH_TO_DBS=$1

PATH_TO_OUTPUT=$2

REMOVE_SET=$3

mkdir -p $PATH_TO_OUTPUT

for f in $PATH_TO_DBS/*.db
do
  if [[ $f == *_modi_covari.db ]] || [[ $f == *_reprocessed.db ]]; then
      #printf 'Skipping "%s"\n' "$f"
      :
  else
      #echo $f
      #python experiment_run_v003.py $f
      #cp $f $(basename $f)
      filename="${f%.*}"
      filename=$(basename $filename)
      new_database_removed_landmarks="${PATH_TO_OUTPUT}/${filename}.db"
      new_database_removed_landmarks_reprocessed="${PATH_TO_OUTPUT}/${filename}_reprocessed.db"
      cp $f $new_database_removed_landmarks
      sqlite3 $new_database_removed_landmarks "Delete from Link where type IN (${REMOVE_SET})"
      rtabmap-reprocess $new_database_removed_landmarks $new_database_removed_landmarks_reprocessed > /dev/null 2>&1
      old=$(rtabmap-report $f | grep -Eo "error lin=([0-9]+,[0-9]+)m" | cut -d= -f2 2>/dev/null)
      new=$(rtabmap-report $new_database_removed_landmarks_reprocessed | grep -Eo "error lin=([0-9]+,[0-9]+)m" | cut -d= -f2 2>/dev/null)
      echo $filename $old $new
  fi
done
