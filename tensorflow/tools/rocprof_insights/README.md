# rocprof-insights 

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
