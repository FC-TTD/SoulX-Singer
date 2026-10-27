import json
import sys
from .api import build_api
from .native_process import load_model, completion, cleanup


def create_app():
    from ttd_model_runtime import Runtime
    from ttd_model_runtime.integrations.fastapi import attach
    runtime = Runtime(load_model, completion=completion, cleanup=cleanup,
        execution_timeout=None)
    return attach(build_api(runtime), runtime=runtime)


def main():
    if sys.argv[1:] == ['describe']:
        class Contract:
            def task(self, function):
                return function
        print(json.dumps(build_api(Contract()).openapi(), ensure_ascii=False))
    elif sys.argv[1:]:
        raise SystemExit('Use python -m hub_runtime [describe]')
    else:
        import uvicorn
        uvicorn.run(create_app(), host='0.0.0.0', port=8000, workers=1)


if __name__ == '__main__':
    main()
