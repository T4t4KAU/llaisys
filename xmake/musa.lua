local musa_home = os.getenv("MUSA_HOME") or "/usr/local/musa"

rule("musa.compile")
    set_extensions(".mu")
    before_buildcmd_file(function (target, batchcmds, sourcefile, opt)
        local objectfile = target:objectfile(sourcefile)
        batchcmds:show_progress(opt.progress, "${color.build.object}compiling.musa %s", sourcefile)
        batchcmds:mkdir(path.directory(objectfile))
        batchcmds:vrunv(path.join(musa_home, "bin", "mcc"), {
            "-c", "-O3", "-fPIC", "-std=c++17",
            "-Iinclude", "-I" .. path.join(musa_home, "include"),
            sourcefile, "-o", objectfile
        })
        table.insert(target:objectfiles(), objectfile)
        batchcmds:add_depfiles(sourcefile)
        batchcmds:set_depmtime(os.mtime(objectfile))
        batchcmds:set_depcache(target:dependfile(objectfile))
    end)
rule_end()
