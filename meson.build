project('tiny_mc', 'c',
        default_options: [ 'c_std=gnu11', 'warning_level=2' ]
)

cc = meson.get_compiler('c')
m_dep = cc.find_library('m')

tiny_mc = executable('tiny_mc',
    'tiny_mc.c',
    dependencies: m_dep
)
