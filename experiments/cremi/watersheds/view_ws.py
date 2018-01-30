import sys
sys.path.append('/home/papec/Work/my_projects/cremi_tools')
from cremi_tools.viewer.bigcat import view


def view_ws():
    view(ram_limit=60)


if __name__ == '__main__':
    view_ws()
