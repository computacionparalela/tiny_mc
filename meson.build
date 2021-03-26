project('tiny_mc', 'c',
        default_options: [ 'c_std=c11', 'warning_level=2' ]
)

cc = meson.get_compiler('c')
m_dep = cc.find_library('m')

tiny_mc = executable('tiny_mc',
    [ 'tiny_mc.c', 'wtime.c' ],
    dependencies: m_dep
)
