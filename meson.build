project('tiny_mc', 'c',
    default_options: [ 'c_std=c11', 'warning_level=2' ]
)

cc = meson.get_compiler('c')
m_dep = cc.find_library('m')
cg_dep = [ m_dep, dependency('glfw3'), dependency('GL'), dependency('GLEW') ]

headless = executable('headless',
    [ 'tiny_mc.c', 'wtime.c', 'photon.c' ],
    dependencies: m_dep
)

head = executable('head',
    [ 'cg_mc.c', 'wtime.c', 'photon.c' ],
    dependencies: cg_dep
)
