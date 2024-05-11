from pathlib import Path
from zensols.pybuild import SetupUtil

su = SetupUtil(
    setup_path=Path(__file__).parent.absolute(),
    name="zensols.calamr",
    package_names=['zensols', 'resources'],
    package_data={'': ['*.conf', '*.json', '*.yml']},
    description='CALAMR: Component ALignment for AMR',
    user='plandes',
    project='calamr',
    keywords=['tooling'],
).setup()
