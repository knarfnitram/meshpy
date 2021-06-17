# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# MeshPy: A beam finite element input generator.
#
# Copyright (c) 2018 Ivo Steinbrecher
#                    Institute for Mathematics and Computer-Based Simulation
#                    Universitaet der Bundeswehr Muenchen
#                    https://www.unibw.de/imcs-en
#
# TODO: Add license.
# -----------------------------------------------------------------------------
"""
Check if the source files in the repository have the correct header.

This file is adapted from LaTeX2AI (https://github.com/stoani89/LaTeX2AI).
"""

# Import python modules.
import os
import subprocess


def get_repository_dir():
    """
    Get the root directory of this repository.
    """

    script_path = os.path.realpath(__file__)
    root_dir = os.path.dirname(os.path.dirname(script_path))
    return root_dir


def get_license_text():
    """
    Return the license text as a string.
    """

    license_path = os.path.join(get_repository_dir(), 'LICENSE')
    with open(license_path) as license_file:
        return license_file.read().strip()


def get_all_source_files():
    """
    Get all source files that should be checked for license headers.
    """

    # Get the files in the git repository.
    repo_dir = get_repository_dir()
    process = subprocess.Popen(['git', 'ls-files'], stdout=subprocess.PIPE,
        cwd=repo_dir)
    out, _err = process.communicate()
    files = out.decode('UTF-8').strip().split('\n')

    source_line_endings = ['.py', '.pyx']
    source_ending_types = {'.py': 'py', '.pyx': 'py'}
    source_files = {'py': []}
    for file in files:
        extension = os.path.splitext(file)[1]
        if extension not in source_line_endings:
            pass
        else:
            source_files[source_ending_types[extension]].append(
                os.path.join(repo_dir, file))
    return source_files


def license_to_source(license_text, source_type):
    """
    Convert the license text to a text that can be written to source code.
    """

    header = None
    start_line = '-' * 77
    if source_type == 'py':
        header = '# -*- coding: utf-8 -*-'
        comment = '#'
    else:
        raise ValueError('Wrong extension!')

    source = []
    if header is not None:
        source.append(header)
    source.append(comment + ' ' + start_line)
    for line in license_text.split('\n'):
        if len(line) > 0:
            source.append(comment + ' ' + line)
        else:
            source.append(comment + line)
    source.append(comment + ' ' + start_line)
    return '\n'.join(source)


def check_license():
    """
    Check the license for all source files.
    """

    license_text = get_license_text()
    source_files = get_all_source_files()

    skip_list = []

    for key in source_files:
        header = license_to_source(license_text, key)
        for file in source_files[key]:
            for skip in skip_list:
                if file.endswith(skip):
                    break
            else:
                with open(file, encoding='ISO-8859-1') as source_file:
                    source_text = source_file.read()
                    if not source_text.startswith(header):
                        print('Wrong header in: {}'.format(file))


if __name__ == '__main__':
    """
    Execution part of script.
    """

    check_license()
