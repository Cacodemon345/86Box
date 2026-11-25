# Rendition Vérité V1000

Extract `PHX49.ROM` from `biosv1-1.ZIP` as `video/verite/phx49.rom` under
an 86Box ROM search path. Select **Rendition Vérité V1000** as the video
card on a PCI machine. The card emulates 4 MiB of VRAM.

The Phoenix ROM runs on the guest x86 CPU. The RISC and pixel engine
come from rust-dos's low-level Vérité implementation. Creative 3D Blaster
PCI CGL loaders require a dump of the card's RISC ROM and are currently
unsupported. Native display timing
uses the programmed CRTC totals and the pixel clock latched through the
V1000's serial PLL port.

The emulated monitor exposes EDID through the VGA connector's DDC1 and DDC2
paths. Native scanout supports Bt485 palette addressing, interlaced fields,
and frame-sequential stereo from the two framebuffer bases.
The DRAM controller accounts for refresh and slow precharge in the RISC
cycle budget using approximate EDO timing.

On x86-64 hosts, the RISC runs cached native blocks for integer and paired
arithmetic, branches, and common local VRAM accesses. The cache checks
instruction bytes before entering a block so microcode changes take effect.
The card's **Dynamic recompiler** option is enabled by default. Turn it off
in the video card settings to run all RISC instructions in the interpreter.
Pixel-engine parameter steps run inline in those blocks. Parameter draws in
common render states run as native spans, compiled per state and cached; each
draw in a block remembers its last span and calls it directly while the state
matches, and the block continues unless the draw wrote into its own code.
Other draws call the C pixel engine and end the block. FIFO and device
operations use the interpreter, which also runs the RISC on other hosts or
when executable memory is unavailable.

The **Run the RISC on its own thread** option, on by default, runs the RISC
and pixel engine on a worker thread. The worker may run up to 1 ms of RISC
time ahead of the guest, or while a DMA runs only as far as the RISC does in
step. The guest's data-FIFO writes queue in front of it, DMA reads guest RAM
directly, and status register reads come from the state the worker last
published; reads of a running DMA first wait for the worker to reach the
guest's time, and other register accesses bring the RISC up to the guest's
time first. The RISC's board writes are applied on the emulation
thread. Turn the option off to run the RISC in step with the guest CPU.
