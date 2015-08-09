import ConfigParser
import sys
from .. import config_parser


def setup_name_maps(config_name, gadget_blocks=False, with_alternates=False):
    name_map = {}
    rev_name_map = {}
    alternates = {}
    try:
        for a, b in config_parser.items(config_name):
            if sys.version_info[0] == 2:
                if gadget_blocks:
                    a = a.upper().ljust(4)
            else:
                if gadget_blocks:
                    a = a.upper().ljust(4).encode("utf-8")
            if b in name_map:
                alternates[name_map[b]] = b
            rev_name_map[a] = b
            name_map[b] = a
    except ConfigParser.NoOptionError:
        pass

    if with_alternates:
        return name_map, rev_name_map, alternates
    else:
        return name_map, rev_name_map


def name_map_function(name_map, rev_name_map):
    def _translate_array_name(name, reverse=False):
        try:
            if reverse:
                return rev_name_map[name]
            else:
                return name_map[name]
        except KeyError:
            return name

    return _translate_array_name


class AdaptiveNameMapper(object):
    def __init__(self, config_name, gadget_blocks=False):
        self._name_map, self._rev_name_map, self._alternates = setup_name_maps(config_name, gadget_blocks,True)

    def _select_alternate_target(self,name):
        self._name_map[self._alternates[name]]=name
        self._rev_name_map[name]=self._alternates[name]

    def _select_alternate_target_if_required(self,name):
        if name not in self._name_map.values() and name in self._alternates:
            self._select_alternate_target(name)

    def __call__(self, name, reverse=False):
        if reverse:
            self._select_alternate_target_if_required(name)
        current_map = self._rev_name_map if reverse else self._name_map
        return current_map.get(name, name)
