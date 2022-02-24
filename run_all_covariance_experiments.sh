#!/bin/bash
PATH_TO_DBS=$1

for f in $PATH_TO_DBS/*.db
do
  if [[ $f == *_modi_covari.db ]] || [[ $f == *_reprocessed.db ]]; then
      printf 'Skipping "%s"\n' "$f"
  else
      python covariance_experiment.py $f
  fi
done
