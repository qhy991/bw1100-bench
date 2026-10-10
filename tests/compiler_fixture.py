"""Small source checkout for preparation tests; never a device Compiler fixture."""


def add_catalog(source, description='fixture'):
    package = source / 'src/open_cake_ir'
    (package / 'compiler').mkdir(parents=True, exist_ok=True)
    (package / 'lab').mkdir(exist_ok=True)
    for path in (package / '__init__.py', package / 'lab/__init__.py'):
        path.write_text('')
    (source / '.gitignore').write_text('__pycache__/\n')
    compiler = package / 'compiler/__init__.py'
    if not compiler.exists():
        compiler.write_text('''import subprocess
class Compiler:
 @classmethod
 def load(cls, root, revision):
  value=cls();value.commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip();return value
''')
    (package / 'compiler/program_passes.py').write_text(
        "from types import SimpleNamespace\nTRANSFORMATIONS=(SimpleNamespace(name='fixture_transform', "
        "parameters=('stage',), description=" + repr(description) + "),)\n")
    (package / 'lab/knowledge.py').write_text('''from open_cake_ir.compiler.program_passes import TRANSFORMATIONS
def transformation_surface(names):
 return [dict(name=x.name,parameters=list(x.parameters),description=x.description) for x in TRANSFORMATIONS if x.name in names]
''')
