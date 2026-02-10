---
name: calkit-pipeline-stage
description: This skill creates a Calkit pipeline stage, which is useful when a process needs to be run in the pipeline inside a given environment.
---

If a pipeline stage is to be created, first we need to choose the `kind`:

- `shell-script`
- `shell-command`
- `python-script`
- `jupyter-notebook`
- `julia-script`
- `julia-command`

Each stage goes in `calkit.yaml`, like:

```yaml
pipeline:
  stages:
    my-stage:
      kind: python-script
      args:
        - --arg1=value1
        - --arg2=value2
        - other-arg
      environment: my-env
      script_path: path/to/script.py
      inputs:
        - input1/file/1.txt
        - input2.csv
      outputs:
        - path: output1
          storage: git
        - path: output2
          storage: dvc
```

The inputs can be detected by reading the script and seeing where files
are read.
Note that the script itself does not need to be listed as an input,
as it is already a dependency of the stage.
An local module or package import would also constitute an input,
as it is a dependency of the stage.
Outputs can be detected by seeing where files are written to.
If the outputs are large or binary, they should be stored in DVC,
otherwise they can be stored in Git.

Each pipeline stage needs an environment defined,
also in `calkit.yaml`:

```yaml
environments:
  my-env:
    kind: conda
    path: path/to/environment.yaml
  my-other-env:
    kind: docker
    path: path/to/Dockerfile
  my-uv-env:
    kind: uv
    path: pyproject.toml
```

To run a stage in the system environment, the `environment`
field can be set to `_system`

Lastly, to check validity of the stage, run:

```sh
calkit check pipeline -c
```

Then, the pipeline (or individual stages) can be run with:

```sh
calkit run
```
