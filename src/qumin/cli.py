import logging
import hydra

from .calc_paradigm_entropy import H_command
from .find_patterns import pat_command
from .find_macroclasses import macroclasses_command
from .make_lattice import lattice_command
from .microclass_heatmap import heatmap_command
from .eval import eval_command
from .utils import Metadata

log = logging.getLogger()

@hydra.main(version_base=None, config_path="config", config_name="qumin")
def qumin_command(cfg):
    log.info(cfg)
    md = Metadata(cfg, __file__)

    if cfg.patterns is None or cfg.action == "patterns":
        overab = cfg.pats.overabundant == False
        for_H = cfg.action == "H"
        for_m = cfg.action == "macroclasses"
        assert overab or not (for_H or for_m), "For this calculation, pats.overabundant must be False"
        assert overab or not for_m, "For this calculation, pats.defective must be False"
        patterns_file = pat_command(cfg, md)
        cfg.patterns = patterns_file

    if cfg.action == "H":
        H_command(cfg, md)
    elif cfg.action == "macroclasses":
        macroclasses_command(cfg, md)
    elif cfg.action == "lattice":
        lattice_command(cfg, md)
    elif cfg.action == "heatmap":
        heatmap_command(cfg, md)
    elif cfg.action == "eval":
        eval_command(cfg, md)

    md.save_metadata()