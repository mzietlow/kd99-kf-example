FROM tensorflow/tfx:0.22.0
COPY ./data /tfx-src/data
COPY ./tfx_utils.py /tfx-src
ENTRYPOINT python3.6 /tfx-src/tfx/scripts/run_executor.py