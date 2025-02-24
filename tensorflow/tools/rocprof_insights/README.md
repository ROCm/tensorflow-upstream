# rocprof-insights 

## introduction
Modern machine learning workloads, such as those built with TensorFlow and JAX, often need to make the most of GPU hardware in terms of both performance and cost-effectiveness. However, balancing compute throughput, memory bandwidth, and parallel execution efficiency can be a complex challenge—especially as models grow in size and computational intensity. To tackle this challenge on AMD GPUs, developers rely on powerful profiling tools such as rocprof, rocprofv2, and the rocprofiler-sdk (rocprofv3) to collect low-level performance data. Interpreting this raw performance data, however, can be time-consuming and non-trivial.

rocprof-insights addresses this gap by providing a streamlined, high-level way to extract and analyze key performance metrics from the output of ROCm profiling tools running from command line frontend. Through automated data parsing, intuitive visualizations, and insightful summaries, rocprof-insights helps developers quickly extract most expensive kernels, memory usage, etc, make informed decisions to guide the optimizeaion of their TensorFlow or JAX workloads. By reducing the complexity of data analysis, rocprof-insights not only accelerates performance tuning but also empowers machine learning practitioners to derive more meaningful insights from their AMD GPU profiling workflows.



## install

```cd rocprof_insights
pip install -e .
```

## running

- on the remote server (docker container)
  ```
  jupyter lab --no-browser --port=8888 --allow-root 
  ```

- on the local terminal

```
ssh -N -f -L localhost:8888:localhost:8888 amd_id@amd_node
```
then from the container running on the server to find something like


To access the server, open this file in a browser:
        `file:///root/.local/share/jupyter/runtime/jpserver-50-open.html`
    Or copy and paste one of these URLs:
        `http://localhost:8888/lab?token=a8d1e5fee5eeae08401b731ce23435702c9fa9191442aef6`
        `http://127.0.0.1:8888/lab?token=a8d1e5fee5eeae08401b731ce23435702c9fa9191442aef6`

copy one of them to a local browser to juyter notebook to work through. 
