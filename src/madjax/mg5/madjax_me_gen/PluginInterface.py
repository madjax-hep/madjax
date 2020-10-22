#####################################################
#                                                   #
#  Source file of the PY8Kernels MG5aMC plugin.     #
#  Use only with consent of its author.             #
#                                                   #
#         author: Valentin Hirschi                  #
#                                                   #
#####################################################

import os
import logging

from madgraph import MadGraph5Error, InvalidCmd
import madgraph.interface.extended_cmd as cmd
import madgraph.iolibs.helas_call_writers as helas_call_writers
import madgraph.interface.madgraph_interface as madgraph_interface
import madjax_me_gen.PluginExporters as PluginExporters

logger = logging.getLogger('madjax_me_gen.Interface')

pjoin = os.path.join


class madjax_me_genPluginInterfaceError(MadGraph5Error):
    """ Error of the Exporter of the madjax_me_gen interface. """


class madjax_me_genPluginInvalidCmd(InvalidCmd):
    """ Invalid command issued to the madjax_me_gen interface. """


class madjax_me_genInterface(madgraph_interface.MadGraphCmd, cmd.CmdShell):
    """Interface for steering the generation/output of madjax_me_gen.
    We make it inherit from CmdShell so that launch_ext_prog does not attempt to start in WebMode."""

    def __init__(self, *args, **opts):
        super(madjax_me_genInterface, self).__init__(*args, **opts)
        self.plugin_output_format_selected = None

    def do_output(self, line):
        """Wrapper to support the syntax output Python <args>.
        This just to add extra freedom in adding special action that may be needed at the output
        stage for these output formats.
        """

        args = self.split_arg(line)
        if len(args) >= 1 and args[0] == 'madjax':
            self.plugin_output_format_selected = 'madjax'
            self.do_output_PythonMEs(' '.join(args[1:]))
        else:
            super(madjax_me_genInterface, self).do_output(' '.join(args))

    def do_output_PythonMEs(self, line):
        args = self.split_arg(line)
        super(madjax_me_genInterface, self).do_output(' '.join(['madjax'] + args))

    def export(self, *args, **opts):
        """Overwrite this so as to force a pythia8 type of output if the output mode is PY8MEs."""

        if self._export_format == 'plugin':
            # Also pass on the aloha model to the exporter (if it has been computed already)
            # so that it will be used when generating the model
            if self.plugin_output_format_selected == 'madjax':
                self._curr_exporter = PluginExporters.PluginProcessExporterPython(
                    self._export_dir,
                    helas_call_writers.PythonUFOHelasCallWriter(self._curr_model),
                )
            else:
                raise MadGraph5Error(
                    "A plugin output format must have been specified at this stage."
                )

        super(madjax_me_genInterface, self).export(*args, **opts)

    # command to change the prompt
    def preloop(self, *args, **opts):
        """only change the prompt after calling  the mother preloop command"""
        super(madjax_me_genInterface, self).preloop(*args, **opts)
        # The colored prompt screws up the terminal for some reason.
        # self.prompt = '\033[92mPY8Kernels > \033[0m'
        self.prompt = 'madjax_me_gen > '
