#!/usr/local/bin/python

# Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
# other Axom Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)

"""
 file: llnl_lc_uberenv_install_tools.py

 description: 
  helpers for installing axom tpls on llnl lc systems.

"""

import os
import socket
import sys
import subprocess
import datetime
import glob
import json
import getpass
import shutil
import time

from os.path import join as pjoin

def sexe(cmd,
         ret_output=False,
         output_file = None,
         echo = False,
         error_prefix = "ERROR:"):
    """ Helper for executing shell commands. """
    if echo:
        print "[exe: %s]" % cmd
    if ret_output:
        p = subprocess.Popen(cmd,
                             shell=True,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT)
        res =p.communicate()[0]
        return p.returncode,res
    elif output_file != None:
        ofile = open(output_file,"w")
        p = subprocess.Popen(cmd,
                             shell=True,
                             stdout= ofile,
                             stderr=subprocess.STDOUT)
        res =p.communicate()[0]
        return p.returncode
    else:
        rcode = subprocess.call(cmd,shell=True)
        if rcode != 0:
            print "[{0} [return code: {1}] from command: {2}]".format(error_prefix, rcode,cmd)
        return rcode


def get_timestamp(t=None,sep="_"):
    """ Creates a timestamp that can easily be included in a filename. """
    if t is None:
        t = datetime.datetime.now()
    sargs = (t.year,t.month,t.day,t.hour,t.minute,t.second)
    sbase = "".join(["%04d",sep,"%02d",sep,"%02d",sep,"%02d",sep,"%02d",sep,"%02d"])
    return  sbase % sargs


def build_info():
    res = {}
    res["built_by"] = os.environ["USER"]
    res["built_from_branch"] = "unknown"
    res["built_from_sha1"]   = "unknown"
    res["platform"] = get_platform()
    rc, out = sexe('git branch -a | grep \"*\"',ret_output=True,error_prefix="WARNING:")
    out = out.strip()
    if rc == 0 and out != "":
        res["built_from_branch"]  = out.split()[1]
    rc,out = sexe('git rev-parse --verify HEAD',ret_output=True,error_prefix="WARNING:")
    out = out.strip()
    if rc == 0 and out != "":
        res["built_from_sha1"] = out
    return res


def write_build_info(ofile):
    print "[build info]"
    binfo_str = json.dumps(build_info(),indent=2)
    print binfo_str
    open(ofile,"w").write(binfo_str)


def log_success(prefix, msg, timestamp=""):
    """
    Called at the end of the process to signal success.
    """
    info = {}
    info["prefix"] = prefix
    info["platform"] = get_platform()
    info["status"] = "success"
    info["message"] = msg
    if timestamp == "":
        info["timestamp"] = get_timestamp()
    else:
        info["timestamp"] = timestamp
    json.dump(info,open(pjoin(prefix,"success.json"),"w"),indent=2)


def log_failure(prefix, msg, timestamp=""):
    """
    Called when the process failed.
    """
    info = {}
    info["prefix"] = prefix
    info["platform"] = get_platform()
    info["status"] = "failed"
    info["message"] = msg
    if timestamp == "":
        info["timestamp"] = get_timestamp()
    else:
        info["timestamp"] = timestamp
    json.dump(info,open(pjoin(prefix,"failed.json"),"w"),indent=2)


def uberenv_create_mirror(prefix, package_name, mirror_path):
    """
    Calls uberenv to create a spack mirror.
    """
    cmd  = "python scripts/uberenv/uberenv.py --prefix=\"{0}\" ".format(prefix)
    cmd += "--package-name={0} ".format(package_name)
    cmd += "--mirror=\"{0}\" --create-mirror ".format(mirror_path)
    res = sexe(cmd, echo=True, error_prefix="WARNING:")
    print "[~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~]"
    print "[The above behavior of 'spack --create-mirror' is normal to throw many warnings]"
    print "[~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~]"
    set_group_and_perms(mirror_path)
    return res


def uberenv_build(prefix, package_name, fullInstall, spec, mirror = None):
    """
    Calls uberenv to install tpls for a given spec to given prefix.
    """
    cmd = "python scripts/uberenv/uberenv.py --prefix=\"{0}\" --spec=\"{1}\" ".format(prefix, spec)
    if not mirror is None:
        cmd += "--mirror=\"{0}\" ".format(mirror)
    if fullInstall:
        cmd += "--install "
    cmd += "--package-name={0} ".format(package_name)
        
    spack_build_log = pjoin(prefix,"output.log.spack.build.%s.txt" % spec.replace(" ", "_"))
    print "[starting spack install of spec %s]" % spec
    print "[log file: %s]" % spack_build_log
    res = sexe(cmd, echo=True, output_file=spack_build_log)
    if res != 0:
        log_failure(prefix, "[ERROR: uberenv/spack build of spec: {0} failed]".format(spec))
    return res


def set_group_and_perms(directory):
    """
    Sets the proper group and access permissions of given input
    directory. 
    """
    print "[changing group and access perms of: %s]" % directory
    # change group to smithdev
    print "[changing group to smithdev]"
    sexe("chgrp -f -R smithdev %s" % (directory),echo=True,error_prefix="WARNING:")
    # change group perms to rwX
    print "[changing perms for smithdev members to rwX]"
    sexe("chmod -f -R g+rwX %s" % (directory),echo=True,error_prefix="WARNING:")
    # change perms for all to rX
    print "[changing perms for all users to rX]"
    sexe("chmod -f -R a+rX %s" % (directory),echo=True,error_prefix="WARNING:")
    print "[done setting perms for: %s]" % directory
    return 0


def build_devtools(builds_dir, timestamp):
    compiler_spec = "%gcc@8.1.0"
    compiler_dir  = "gcc-8.1.0"
    print "[Building devtools using compiler spec: {0}]".format(compiler_spec)

    # unique install location
    prefix = pjoin(builds_dir, timestamp)
    if not os.path.exists(prefix):
        os.makedirs(prefix)

    # Use shared mirror
    mirror_dir = get_shared_tpl_mirror_dir()
    print "[Using mirror location: {0}]".format(mirror_dir)
    uberenv_create_mirror(prefix, "serac_devtools", mirror_dir)

    # write info about this build
    write_build_info(pjoin(prefix,"info.json"))

    # use uberenv to install devtools
    start_time = time.time()
    res = uberenv_build(prefix, "serac_devtools", True, compiler_spec, mirror_dir)
    end_time = time.time()

    print "[Build time: {0}]".format(convertSecondsToReadableTime(end_time - start_time))
    if res != 0:
        print "[ERROR: Failed build of devtools for spec %s]\n" % compiler_spec
    else:
        # Only update the latest symlink if successful
        link_path = pjoin(builds_dir, "latest")
        install_dir = pjoin(prefix, compiler_dir)
        print "[Creating symlink to latest devtools build:\n{0}\n->\n{1}]".format(link_path, install_dir)
        if os.path.exists(link_path):
            if not os.path.islink(link_path):
                print "[ERROR: Latest devtools link path exists and is not a link: {0}".format(link_path)
                return 1
            os.unlink(link_path)
        os.symlink(install_dir, link_path)

        # Clean up directories we don't need to save
        dir_names = ["builds", "spack"]
        for dir_name in dir_names:
            path_to_be_deleted = pjoin(prefix, dir_name)
            print "[Removing path after successful devtools build: {0}]".format(path_to_be_deleted)
            if os.path.exists(path_to_be_deleted):
                shutil.rmtree(path_to_be_deleted)
        link_to_be_deleted = pjoin(prefix, "serac_devtools-install")
        print "[Removing link after successful devtools build: {0}]".format(link_to_be_deleted)
        if os.path.exists(link_to_be_deleted):
            os.unlink(link_to_be_deleted)

        print "[SUCCESS: Finished build devtools for spec %s]\n" % compiler_spec

    # set proper perms for installed devtools
    set_group_and_perms(prefix)

    return res


def get_specs_for_current_machine():
    repo_dir = get_repo_dir()
    specs_json_path = pjoin(repo_dir, "scripts/uberenv/specs.json")

    with open(specs_json_path, 'r') as f:
        specs_json = json.load(f)

    sys_type = get_system_type()
    machine_name = get_machine_name()

    specs = []
    if machine_name in specs_json.keys():
        specs = specs_json[machine_name]
    else:
        specs = specs_json[sys_type]

    specs = ['%' + spec for spec in specs]

    return specs


def get_host_configs_for_current_machine(src_dir):
    host_configs = []

    # Note: This function is called in two situations:
    # (1) To test the checked-in host-configs from a source dir 
    #   In that case, check the 'host-configs' directory
    # (2) To test the uberenv-generated host-configs
    #   In that case, host-configs should be in src_dir
    
    host_configs_dir = pjoin(src_dir, "host-configs")
    if not os.path.isdir(host_configs_dir):
        host_configs_dir = src_dir

    hostname_base = get_machine_name()

    host_configs = glob.glob(pjoin(host_configs_dir, hostname_base + "*.cmake"))

    return host_configs


def get_host_config_root(host_config):
    return os.path.splitext(os.path.basename(host_config))[0]


def get_build_dir(prefix, host_config):
    host_config_root = get_host_config_root(host_config)
    return pjoin(prefix, "build-" + host_config_root)


def get_repo_dir():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.abspath(pjoin(script_dir, "../.."))

def get_build_and_test_root(prefix, timestamp):
    return pjoin(prefix,"_axom_build_and_test_%s" % timestamp)


def get_machine_name():
    return socket.gethostname().rstrip('1234567890')


def get_system_type():
    return os.environ["SYS_TYPE"]


def get_platform():
    return get_system_type() if "SYS_TYPE" in os.environ else get_machine_name()


def get_username():
    return getpass.getuser()


def get_shared_base_dir():
    return "/usr/WS2/smithdev"


def get_shared_devtool_dir():
    dir = pjoin(get_shared_base_dir(), "devtools")
    dir = pjoin(dir, get_system_type())
    return dir


def get_shared_tpl_mirror_dir():
    return pjoin(get_shared_base_dir(), "mirror")


def get_shared_libraries_dir():
    return pjoin(get_shared_tpl_base_dir(), "libraries")


def on_rz():
    machine_name = get_machine_name()
    if machine_name.startswith("rz"):
        return True
    return False


def convertSecondsToReadableTime(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)

