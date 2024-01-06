#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <wchar.h>
#include <stdatomic.h>
#define HAVE_STDARG_H
#include <86box/86box.h>
#include <86box/device.h>
#include <86box/io.h>
#include <86box/mem.h>
#include <86box/timer.h>
#include <86box/pci.h>
#include <86box/rom.h>
#include <86box/plat.h>
#include <86box/thread.h>
#include <86box/video.h>
#include <86box/i2c.h>
#include <86box/vid_ddc.h>
#include <86box/vid_svga.h>
#include <86box/vid_svga_render.h>

static video_timings_t timing_banshee     = { .type = VIDEO_PCI, .write_b = 2, .write_w = 2, .write_l = 1, .read_b = 20, .read_w = 20, .read_l = 21 };

static uint32_t be_to_le(uint32_t be_data)
{
	return (be_data >> 24) | ((be_data >> 8) & 0xff00) | ((be_data << 8) & 0xff0000) | (be_data << 24);
}

#define OPCODE_IMM_MASK 0xffff
#define OPCODE_D_SHIFT 16
#define OPCODE_D_MASK (0xff << OPCODE_D_SHIFT)
#define OPCODE_S2_SHIFT 8
#define OPCODE_S2_MASK (0xff << OPCODE_S2_SHIFT)
#define OPCODE_S1_SHIFT 0
#define OPCODE_S1_MASK (0xff << OPCODE_S1_SHIFT)
#define OPCODE_INT_IMM_MASK OPCODE_S1_MASK

#define OPCODE_STORE_OFFSET_SHIFT 16
#define OPCODE_STORE_OFFSET_MASK (0xff << OPCODE_STORE_OFFSET_SHIFT)
#define OPCODE_LOAD_OFFSET_SHIFT 8
#define OPCODE_LOAD_OFFSET_MASK (0xff << OPCODE_LOAD_OFFSET_SHIFT)

#define OPCODE_BRANCH_OFFSET_SHIFT 8
#define OPCODE_BRANCH_OFFSET_MASK (0xffff << OPCODE_BRANCH_OFFSET_SHIFT)

#define OPCODE_SPRII_OFFSET_SHIFT 8
#define OPCODE_SPRII_OFFSET_MASK (0xffff << OPCODE_BRANCH_OFFSET_SHIFT)

enum op_types
{
	OP_TYPE_INTEGER,
	OP_TYPE_INTEGER_IMM,
	OP_TYPE_LOAD,
	OP_TYPE_STORE,
	OP_TYPE_SORT,
	OP_TYPE_LOAD_IMM,
	OP_TYPE_SPRII,
	OP_TYPE_BRANCH,
	OP_TYPE_JUMP_ABS,
	OP_TYPE_JUMP_REL,
	OP_TYPE_OTHER
} op_types;

static struct
{
	const char *name;
	enum op_types op_type;
	int offset_mul;
} opcodes[256] =
{
	[0x00] = { .name = "add", .op_type = OP_TYPE_INTEGER_IMM },
	[0x01] = { .name = "sub", .op_type = OP_TYPE_INTEGER_IMM },
	[0x02] = { .name = "andn", .op_type = OP_TYPE_INTEGER_IMM },
	[0x03] = { .name = "subf", .op_type = OP_TYPE_INTEGER_IMM },
	[0x04] = { .name = "and", .op_type = OP_TYPE_INTEGER_IMM },
	[0x05] = { .name = "or", .op_type = OP_TYPE_INTEGER_IMM },
	[0x06] = { .name = "nor", .op_type = OP_TYPE_INTEGER_IMM },
	[0x07] = { .name = "xor", .op_type = OP_TYPE_INTEGER_IMM },
	[0x08] = { .name = "mul", .op_type = OP_TYPE_INTEGER_IMM },
	[0x09] = { .name = "mulsr8", .op_type = OP_TYPE_INTEGER_IMM },
	[0x0a] = { .name = "mulsr16", .op_type = OP_TYPE_INTEGER_IMM },
	[0x0b] = { .name = "mulsr24", .op_type = OP_TYPE_INTEGER_IMM },
	[0x0c] = { .name = "mulsr32", .op_type = OP_TYPE_INTEGER_IMM },
	[0x0d] = { .name = "mulsr31", .op_type = OP_TYPE_INTEGER_IMM },
	[0x0e] = { .name = "clip", .op_type = OP_TYPE_OTHER },
	[0x0f] = { .name = "f2i", .op_type = OP_TYPE_INTEGER_IMM },

	[0x10] = { .name = "add", .op_type = OP_TYPE_INTEGER },
	[0x11] = { .name = "sub", .op_type = OP_TYPE_INTEGER },
	[0x12] = { .name = "andn", .op_type = OP_TYPE_INTEGER },
	[0x13] = { .name = "subf", .op_type = OP_TYPE_INTEGER },
	[0x14] = { .name = "and", .op_type = OP_TYPE_INTEGER },
	[0x15] = { .name = "or", .op_type = OP_TYPE_INTEGER },
	[0x16] = { .name = "nor", .op_type = OP_TYPE_INTEGER },
	[0x17] = { .name = "xor", .op_type = OP_TYPE_INTEGER },
	[0x18] = { .name = "mul", .op_type = OP_TYPE_INTEGER },
	[0x19] = { .name = "mulsr8", .op_type = OP_TYPE_INTEGER },
	[0x1a] = { .name = "mulsr16", .op_type = OP_TYPE_INTEGER },
	[0x1b] = { .name = "mulsr24", .op_type = OP_TYPE_INTEGER },
	[0x1c] = { .name = "mulsr32", .op_type = OP_TYPE_INTEGER },
	[0x1d] = { .name = "mulsr31", .op_type = OP_TYPE_INTEGER },
	[0x1e] = { .name = "clip", .op_type = OP_TYPE_OTHER },
	[0x1f] = { .name = "f2i", .op_type = OP_TYPE_INTEGER },

	[0x20] = { .name = "add.v", .op_type = OP_TYPE_INTEGER },
	[0x21] = { .name = "sub.v", .op_type = OP_TYPE_INTEGER },
	[0x22] = { .name = "andn.v", .op_type = OP_TYPE_INTEGER },
	[0x23] = { .name = "subf.v", .op_type = OP_TYPE_INTEGER },
	[0x24] = { .name = "and.v", .op_type = OP_TYPE_INTEGER },
	[0x25] = { .name = "or.v", .op_type = OP_TYPE_INTEGER },
	[0x26] = { .name = "nor.v", .op_type = OP_TYPE_INTEGER },
	[0x27] = { .name = "xor.v", .op_type = OP_TYPE_INTEGER },
	[0x28] = { .name = "mul.v", .op_type = OP_TYPE_INTEGER },
	[0x29] = { .name = "mulsr8.v", .op_type = OP_TYPE_INTEGER },
	[0x2a] = { .name = "mulsr16.v", .op_type = OP_TYPE_INTEGER },
	[0x2b] = { .name = "mulsr24.v", .op_type = OP_TYPE_INTEGER },
	[0x2c] = { .name = "mulsr32.v", .op_type = OP_TYPE_INTEGER },
	[0x2d] = { .name = "mulsr31.v", .op_type = OP_TYPE_INTEGER },
	[0x2e] = { .name = "clip.v", .op_type = OP_TYPE_OTHER },

	[0x30] = { .name = "add.vs", .op_type = OP_TYPE_INTEGER },
	[0x31] = { .name = "sub.vs", .op_type = OP_TYPE_INTEGER },
	[0x32] = { .name = "andn.vs", .op_type = OP_TYPE_INTEGER },
	[0x33] = { .name = "subf.vs", .op_type = OP_TYPE_INTEGER },
	[0x34] = { .name = "and.vs", .op_type = OP_TYPE_INTEGER },
	[0x35] = { .name = "or.vs", .op_type = OP_TYPE_INTEGER },
	[0x36] = { .name = "nor.vs", .op_type = OP_TYPE_INTEGER },
	[0x37] = { .name = "xor.vs", .op_type = OP_TYPE_INTEGER },
	[0x38] = { .name = "mul.vs", .op_type = OP_TYPE_INTEGER },
	[0x39] = { .name = "mulsr8.vs", .op_type = OP_TYPE_INTEGER },
	[0x3a] = { .name = "mulsr16.vs", .op_type = OP_TYPE_INTEGER },
	[0x3b] = { .name = "mulsr24.vs", .op_type = OP_TYPE_INTEGER },
	[0x3c] = { .name = "mulsr32.vs", .op_type = OP_TYPE_INTEGER },
	[0x3d] = { .name = "mulsr31.vs", .op_type = OP_TYPE_INTEGER },
	[0x3e] = { .name = "clip.vs", .op_type = OP_TYPE_OTHER },

	[0x40] = { .name = "addif", .op_type = OP_TYPE_OTHER },
	[0x41] = { .name = "subif", .op_type = OP_TYPE_OTHER },
	[0x42] = { .name = "seedsr", .op_type = OP_TYPE_INTEGER_IMM },
	[0x43] = { .name = "subfif", .op_type = OP_TYPE_INTEGER_IMM },
	[0x44] = { .name = "rotr", .op_type = OP_TYPE_INTEGER_IMM },
	[0x45] = { .name = "sl", .op_type = OP_TYPE_INTEGER_IMM },
	[0x46] = { .name = "asr", .op_type = OP_TYPE_INTEGER_IMM },
	[0x47] = { .name = "sr", .op_type = OP_TYPE_INTEGER_IMM },
	[0x48] = { .name = "slt", .op_type = OP_TYPE_INTEGER_IMM },
	[0x49] = { .name = "sltu", .op_type = OP_TYPE_INTEGER_IMM },
	[0x4a] = { .name = "seq", .op_type = OP_TYPE_INTEGER_IMM },
	[0x4b] = { .name = "addsl8", .op_type = OP_TYPE_INTEGER_IMM },
	[0x4c] = { .name = "min", .op_type = OP_TYPE_INTEGER_IMM },
	[0x4d] = { .name = "max", .op_type = OP_TYPE_INTEGER_IMM },
	[0x4e] = { .name = "sprii", .op_type = OP_TYPE_SPRII },
	[0x4f] = { .name = "spri", .op_type = OP_TYPE_OTHER },

	[0x52] = { .name = "abs", .op_type = OP_TYPE_OTHER },
	[0x53] = { .name = "seedsr", .op_type = OP_TYPE_INTEGER },
	[0x54] = { .name = "rotr", .op_type = OP_TYPE_INTEGER },
	[0x55] = { .name = "sl", .op_type = OP_TYPE_INTEGER },
	[0x56] = { .name = "asr", .op_type = OP_TYPE_INTEGER },
	[0x57] = { .name = "sr", .op_type = OP_TYPE_INTEGER },
	[0x58] = { .name = "slt", .op_type = OP_TYPE_INTEGER },
	[0x59] = { .name = "sltu", .op_type = OP_TYPE_INTEGER },
	[0x5a] = { .name = "seq", .op_type = OP_TYPE_INTEGER },
	[0x5c] = { .name = "min", .op_type = OP_TYPE_INTEGER },
	[0x5d] = { .name = "max", .op_type = OP_TYPE_INTEGER },
	[0x5e] = { .name = "at", .op_type = OP_TYPE_INTEGER },
	[0x5f] = { .name = "spr", .op_type = OP_TYPE_OTHER },

	[0x60] = { .name = "bez", .op_type = OP_TYPE_BRANCH },
	[0x61] = { .name = "bnez", .op_type = OP_TYPE_BRANCH },
	[0x62] = { .name = "bgez", .op_type = OP_TYPE_BRANCH },
	[0x63] = { .name = "blz", .op_type = OP_TYPE_BRANCH },
	[0x64] = { .name = "bgz", .op_type = OP_TYPE_BRANCH },
	[0x65] = { .name = "blez", .op_type = OP_TYPE_BRANCH },
	[0x68] = { .name = "rjmp", .op_type = OP_TYPE_OTHER },
	[0x6a] = { .name = "rjmpl", .op_type = OP_TYPE_OTHER },
	[0x6b] = { .name = "getpc", .op_type = OP_TYPE_OTHER },
	[0x6c] = { .name = "jmp", .op_type = OP_TYPE_OTHER },
	[0x6d] = { .name = "halt", .op_type = OP_TYPE_OTHER },
	[0x6e] = { .name = "jmpl", .op_type = OP_TYPE_OTHER },
	[0x6f] = { .name = "jmprl", .op_type = OP_TYPE_OTHER },

	[0x70] = { .name = "lb", .op_type = OP_TYPE_LOAD, .offset_mul = 1 },
	[0x71] = { .name = "lh", .op_type = OP_TYPE_LOAD, .offset_mul = 2 },
	[0x72] = { .name = "lw", .op_type = OP_TYPE_LOAD, .offset_mul = 4 },
	[0x73] = { .name = "pre", .op_type = OP_TYPE_OTHER, .offset_mul = 8 },
	[0x74] = { .name = "lv", .op_type = OP_TYPE_LOAD, .offset_mul = 8 },
	[0x75] = { .name = "lvra", .op_type = OP_TYPE_LOAD, .offset_mul = 8 },
	[0x76] = { .name = "li", .op_type = OP_TYPE_LOAD_IMM },
	[0x77] = { .name = "lui", .op_type = OP_TYPE_LOAD_IMM },
	[0x78] = { .name = "sb", .op_type = OP_TYPE_STORE, .offset_mul = 1 },
	[0x79] = { .name = "sh", .op_type = OP_TYPE_STORE, .offset_mul = 2 },
	[0x7a] = { .name = "sw", .op_type = OP_TYPE_STORE, .offset_mul = 4 },
	[0x7c] = { .name = "sv", .op_type = OP_TYPE_STORE, .offset_mul = 8 },
	[0x7d] = { .name = "sy", .op_type = OP_TYPE_STORE, .offset_mul = 8 },
	[0x7e] = { .name = "scr", .op_type = OP_TYPE_STORE, .offset_mul = 8 },
	[0x7f] = { .name = "scb", .op_type = OP_TYPE_STORE, .offset_mul = 8 },

	[0xc0] = { .name = "getxy", .op_type = OP_TYPE_OTHER },
	[0xc1] = { .name = "getyx", .op_type = OP_TYPE_OTHER },
	[0xc2] = { .name = "getra", .op_type = OP_TYPE_OTHER },
	[0xc3] = { .name = "getgb", .op_type = OP_TYPE_OTHER },
	[0xc4] = { .name = "sort", .op_type = OP_TYPE_INTEGER },

	[0xd0] = { .name = "step x/y", .op_type = OP_TYPE_OTHER },
	[0xd1] = { .name = "step q", .op_type = OP_TYPE_OTHER },
	[0xd2] = { .name = "step r/b", .op_type = OP_TYPE_OTHER },
	[0xd3] = { .name = "step i/v", .op_type = OP_TYPE_OTHER },
	[0xd4] = { .name = "step f", .op_type = OP_TYPE_OTHER },
	[0xd5] = { .name = "step g/b", .op_type = OP_TYPE_OTHER },
	[0xd6] = { .name = "step z/a", .op_type = OP_TYPE_OTHER },
	[0xd7] = { .name = "step cnt", .op_type = OP_TYPE_OTHER },
	[0xd8] = { .name = "step xy", .op_type = OP_TYPE_OTHER },
	[0xd9] = { .name = "step qu", .op_type = OP_TYPE_OTHER },
	[0xda] = { .name = "step rv", .op_type = OP_TYPE_OTHER },
	[0xdb] = { .name = "step iv", .op_type = OP_TYPE_OTHER },
	[0xdc] = { .name = "step zv", .op_type = OP_TYPE_OTHER },
	[0xdd] = { .name = "step gb", .op_type = OP_TYPE_OTHER },
	[0xde] = { .name = "step za", .op_type = OP_TYPE_OTHER },
	[0xdf] = { .name = "step cnt", .op_type = OP_TYPE_OTHER },

	[0xe0] = { .name = "drawp", .op_type = OP_TYPE_INTEGER },
	[0xe1] = { .name = "rdrawp", .op_type = OP_TYPE_INTEGER },
	[0xe4] = { .name = "drawpxy", .op_type = OP_TYPE_INTEGER },
	[0xe5] = { .name = "rdrawpxy", .op_type = OP_TYPE_INTEGER },

	[0xe7] = { .name = "mbltfo", .op_type = OP_TYPE_INTEGER },
	[0xe8] = { .name = "fill", .op_type = OP_TYPE_INTEGER },
	[0xe9] = { .name = "mbrush", .op_type = OP_TYPE_INTEGER },
	[0xea] = { .name = "mblt", .op_type = OP_TYPE_INTEGER },
	[0xeb] = { .name = "mbltf", .op_type = OP_TYPE_INTEGER },
	[0xec] = { .name = "cbrush8", .op_type = OP_TYPE_INTEGER },
	[0xed] = { .name = "cbrush16", .op_type = OP_TYPE_INTEGER },
	[0xee] = { .name = "cbrush32", .op_type = OP_TYPE_INTEGER },
	[0xef] = { .name = "cbltf", .op_type = OP_TYPE_INTEGER },

	[0xf0] = { .name = "draw1", .op_type = OP_TYPE_INTEGER },
	[0xf1] = { .name = "rdraw1", .op_type = OP_TYPE_INTEGER },
	[0xf2] = { .name = "draw2", .op_type = OP_TYPE_INTEGER },
	[0xf3] = { .name = "rdraw2", .op_type = OP_TYPE_INTEGER },
	[0xf8] = { .name = "draw3", .op_type = OP_TYPE_INTEGER },
	[0xf9] = { .name = "rdraw3", .op_type = OP_TYPE_INTEGER },
	[0xfa] = { .name = "draw4", .op_type = OP_TYPE_INTEGER },
	[0xfb] = { .name = "rdraw4", .op_type = OP_TYPE_INTEGER },
	[0xfc] = { .name = "draw5", .op_type = OP_TYPE_INTEGER },
	[0xfd] = { .name = "rdraw5", .op_type = OP_TYPE_INTEGER },
	[0xfe] = { .name = "tri", .op_type = OP_TYPE_INTEGER },
};

static const char *register_names[256] =
{
	[255] = "fp",
	[254] = "ra",
	[253] = "ira",
	[252] = "sp",

	[57] = "msk11",
	[56] = "msk12",

	[47] = "riscintr",
	[46] = "winclip",

	[41] = "npfifo",
	[40] = "npswfifo",
	[39] = "timer",
	[38] = "clip",
	[37] = "flag",
	[36] = "scale",
	[35] = "seg",
	[34] = "seed",
	[33] = "fifo",
	[32] = "swfifo",
	[31] = "f800",
	[30] = "f000",
	[29] = "e800",
	[28] = "e000",
	[27] = "d800",
	[26] = "d000",
	[25] = "c800",
	[24] = "c000",
	[23] = "b800",
	[22] = "b000",
	[21] = "bit17",
	[20] = "bit23",
	[19] = "bit25",
	[18] = "fpfrac",
	[17] = "bit16",
	[16] = "allfs",
	[15] = "if1_0",
	[14] = "zer3",
	[13] = "minus1",
	[12] = "zer2",
	[11] = "b1b3msk",
	[10] = "b0b2msk",
	[9] = "h1msk",
	[8] = "h0msk",
	[7] = "b3msk",
	[6] = "b2msk",
	[5] = "b1msk",
	[4] = "b0msk",
	[3] = "msb",
	[2] = "if0_5",
	[1] = "zer1",
	[0] = "zero"
};

static void get_register_name(char *dest, uint8_t reg)
{
	if (register_names[reg])
		strcpy(dest, register_names[reg]);
	else
		sprintf(dest, "%%%i", reg);
}

static void get_dest_register_name(char *dest, uint32_t opcode)
{
	int reg = (opcode & OPCODE_D_MASK) >> OPCODE_D_SHIFT;
	if (reg == 40)
	{
		/*40 is different registers for read/write*/
		strcpy(dest, "drawctl");
	}
	else
		get_register_name(dest, (opcode & OPCODE_D_MASK) >> OPCODE_D_SHIFT);
}
static void get_s2_register_name(char *dest, uint32_t opcode)
{
	get_register_name(dest, (opcode & OPCODE_S2_MASK) >> OPCODE_S2_SHIFT);
}
static void get_s1_register_name(char *dest, uint32_t opcode)
{
	get_register_name(dest, (opcode & OPCODE_S1_MASK) >> OPCODE_S1_SHIFT);
}

static const char *pixel_register_names[256] =
{
	[0] = "SrcBase",
	[1] = "SrcMode",
	[2] = "SrcMask",
	[3] = "Stride",
	[4] = "DstBase",
	[5] = "DstMode",
	[6] = "DstFmt",
	[7] = "PMask",
	[8] = "Pattern",
	[10] = "IPat",
	[12] = "PatMode",
	[13] = "PatOffset",
	[14] = "ScissorX",
	[15] = "ScissorY",
	[16] = "ZBase",
	[17] = "DitherOffset",
	[19] = "FGColor",
	[20] = "BGColor",
	[21] = "FogColor",
	[22] = "DstColor",
	[23] = "ChromaColor",
	[24] = "YUVContrast",
	[26] = "ChromaMask",
	[27] = "FGColorRGB",
	[28] = "Pick",
	[29] = "AlphaThres",
	[30] = "SpecColor",
	[31] = "Sync",
	[32] = "Tex4Pal[0]",
	[33] = "Tex4Pal[1]",
	[34] = "Tex4Pal[2]",
	[35] = "Tex4Pal[3]",
	[36] = "Tex4Pal[4]",
	[37] = "Tex4Pal[5]",
	[38] = "Tex4Pal[6]",
	[39] = "Tex4Pal[7]",

	[48] = "SrcFmt",
	[49] = "SrcFunc",
	[50] = "SrcFilter",
	[51] = "SrcColorNoPad",
	[52] = "SwapUV",
	[53] = "ChromaKey",
	[54] = "ZScissorEn",
	[55] = "SrcBGR",
	[56] = "UMask",
	[57] = "VMask",
	[58] = "SrcStride",
	[59] = "DstStride",
	[60] = "ZStride",
	[61] = "ChromaBlackYUV",
	[62] = "UClamp",
	[63] = "VClamp",
	[64] = "ALUMode",
	[65] = "BlendSrcFunc",
	[66] = "BlendDstFunc",
	[67] = "ZBufMode",
	[68] = "ZBufWrMode",
	[69] = "YUV2RGB",
	[70] = "BlendEnable",
	[71] = "DitherEnable",
	[72] = "FogEnable",
	[73] = "DstColorNoPad",
	[74] = "DstRdDisable",
	[75] = "DstBGR",
	[76] = "TranspReject",
	[80] = "PatLengthM1",
	[81] = "PatEnable",
	[82] = "PatOpaque",
	[83] = "SpecularEn"

};

#pragma pack(push, 1)
typedef struct v2200_t
{
    svga_t svga;
    uint8_t pci_regs[256];
	union
	{
		uint8_t regs[256];
		uint16_t regs_w[128];
		uint32_t regs_l[64];
	};

    mem_mapping_t linear_mapping;
    mem_mapping_t reg_mapping;
    rom_t bios_rom;
    uint32_t linear_base, reg_base;
    uint32_t io_base;

    /* RISC regs */
    uint32_t risc_pc;
    uint8_t slot;
} v2200_t;
#pragma pack(pop)

static void get_pixel_register_name(char *dest, uint32_t opcode)
{
	int reg = (opcode & OPCODE_S1_MASK) >> OPCODE_S1_SHIFT;

	if (pixel_register_names[reg])
		strcpy(dest, pixel_register_names[reg]);
	else
		sprintf(dest, "%%%i", reg);
}

static void disassemble_opcode(uint32_t opcode, uint32_t addr)
{
	char pixel_reg_name[16];
	char dest_reg_name[16];
	char s2_reg_name[16];
	char s1_reg_name[16];
	uint8_t op = opcode >> 24;
	int s2 = (opcode & OPCODE_S2_MASK) >> OPCODE_S2_SHIFT;
	int s1 = (opcode & OPCODE_S1_MASK) >> OPCODE_S1_SHIFT;
	int32_t offset16 = (int32_t)(int16_t)((opcode & OPCODE_BRANCH_OFFSET_MASK) >> OPCODE_BRANCH_OFFSET_SHIFT);

	get_dest_register_name(dest_reg_name, opcode);
	get_s2_register_name(s2_reg_name, opcode);
	get_s1_register_name(s1_reg_name, opcode);

	if (!opcodes[op].name)
	{
		printf("<undef>");
		return;
	}
	
	if (!opcode)
	{
		printf("nop");
		return;
	}

	switch (opcodes[op].op_type)
	{
		case OP_TYPE_INTEGER:
		printf("%s\t%s, %s, %s", opcodes[op].name, dest_reg_name, s2_reg_name, s1_reg_name);
		break;

		case OP_TYPE_INTEGER_IMM:
		printf("%s\t%s, %s, #0x%x", opcodes[op].name, dest_reg_name, s2_reg_name, opcode & OPCODE_INT_IMM_MASK);
		break;

		case OP_TYPE_LOAD:
		printf("%s\t%s, %x(%s)", opcodes[op].name, dest_reg_name,
					((opcode & OPCODE_LOAD_OFFSET_MASK) >> OPCODE_LOAD_OFFSET_SHIFT) * opcodes[op].offset_mul,
					s1_reg_name);
		break;

		case OP_TYPE_STORE:
		printf("%s\t%x(%s), %s", opcodes[op].name,
					((opcode & OPCODE_STORE_OFFSET_MASK) >> OPCODE_STORE_OFFSET_SHIFT) * opcodes[op].offset_mul,
					s1_reg_name, s2_reg_name);
		break;

		case OP_TYPE_LOAD_IMM:
		printf("%s\t%s, #0x%x", opcodes[op].name, dest_reg_name, opcode & OPCODE_IMM_MASK);
		break;

		case OP_TYPE_SPRII:
		get_pixel_register_name(pixel_reg_name, opcode);
		printf("%s\t%s, 0x%04x", opcodes[op].name, pixel_reg_name, (opcode & OPCODE_SPRII_OFFSET_MASK) >> OPCODE_SPRII_OFFSET_SHIFT);
		break;

		case OP_TYPE_BRANCH:
		printf("%s\t%s, 0x%x", opcodes[op].name, s1_reg_name, (addr + 4 + offset16*4) & 0xffffff);
		break;

		case OP_TYPE_OTHER:
		switch (op)
		{
			case 0x0e: /*clip - S2 and S1 registers*/
			printf("clip\t%s, %s", s2_reg_name, opcode & OPCODE_INT_IMM_MASK);
			break;

			case 0x1e: case 0x2e: case 0x3e: /*clip - S2 and S1 registers*/
			printf("%s\t%s, %s", opcodes[op].name, s2_reg_name, s1_reg_name);
			break;

			case 0x40: case 0x41 : /*addif / subif - immediate shifted 16*/
			printf("%s\t%s, %s, #0x%x", opcodes[op].name, dest_reg_name, s2_reg_name, (opcode & OPCODE_INT_IMM_MASK) << 16);
			break;

			case 0x4f: /*spri - S1 = pixel register, S2 = source*/
			get_pixel_register_name(pixel_reg_name, opcode);
			printf("spri\t%s, %s", pixel_reg_name, s2_reg_name);
			break;

			case 0x52: /*abs - D and S1 registers*/
			case 0xc0: case 0xc1: case 0xc2: case 0xc3: /*getxy / getyx / getrg / getba - D and S1 registers*/
			printf("%s\t%s, %s", opcodes[op].name, dest_reg_name, s1_reg_name);
			break;

			case 0x5f: /*spr - S1 = indirect pixel register, S2 = source*/
			printf("spr\t(%s), %s", s1_reg_name, s2_reg_name);
			break;

			case 0x6b: /*getpc - D register but no other parameters*/
			printf("getpc\t%s", dest_reg_name);
			break;

			case 0x68: case 0x6a: /*rjmp / rjmpl - relative 24-bit address*/
			printf("%s\t0x%06x", opcodes[op].name, (addr + 4 + (opcode & 0x3fffff) * 4) & 0xffffff);
			break;

			case 0x6c: case 0x6e: /*jmp / jmpl - absolute 24-bit address*/
			printf("%s\t0x%06x", opcodes[op].name, (opcode & 0x3fffff) << 2);
			break;

			case 0x6d: /*halt*/
			printf("halt\t0x%04x", opcode & 0xffff);
			break;

			case 0x6f: /*jmprl - D and S1 registers*/
			printf("jmprl\t%s, (%s)", dest_reg_name, s1_reg_name);
			break;

			case 0x73: /*pre - load instruction without D register*/
			printf("%s\t%x(%s)", opcodes[op].name,
						((opcode & OPCODE_LOAD_OFFSET_MASK) >> OPCODE_LOAD_OFFSET_SHIFT) * opcodes[op].offset_mul,
						s1_reg_name);
			break;

			case 0xd0: /*step x/y*/
			if (!(s2 & 1))
				printf("step\tx, %s, %s, %s", dest_reg_name, s2_reg_name, s1_reg_name);
			else
				printf("step\ty, %s, %s, %s", dest_reg_name, s2_reg_name, s1_reg_name);
			break;

			case 0xd1: /*step q*/
			printf("step\tq, %s, %s, %s", dest_reg_name, s2_reg_name, s1_reg_name);
			break;

			case 0xd2: /*step r/v*/
			if (!(s2 & 1))
				printf("step\tr, %s, %s, %s", dest_reg_name, s2_reg_name, s1_reg_name);
			else
				printf("step\tv, %s, %s, %s", dest_reg_name, s2_reg_name, s1_reg_name);
			break;

			case 0xd3: /*step i/u*/
			if (!(s2 & 1))
				printf("step\ti, %s, %s, %s", dest_reg_name, s2_reg_name, s1_reg_name);
			else
				printf("step\tu, %s, %s, %s", dest_reg_name, s2_reg_name, s1_reg_name);
			break;

			case 0xd4: /*step f*/
			printf("step\tf, %s, %s, %s", dest_reg_name, s2_reg_name, s1_reg_name);
			break;

			case 0xd5: /*step g/b*/
			if (!(s2 & 1))
				printf("step\tg, %s, %s, %s", dest_reg_name, s2_reg_name, s1_reg_name);
			else
				printf("step\tb, %s, %s, %s", dest_reg_name, s2_reg_name, s1_reg_name);
			break;

			case 0xd6: /*step z/a*/
			if (!(s2 & 1))
				printf("step\tz, %s, %s, %s", dest_reg_name, s2_reg_name, s1_reg_name);
			else
				printf("step\ta, %s, %s, %s", dest_reg_name, s2_reg_name, s1_reg_name);
			break;

			case 0xd7: /*step cnt*/
			printf("step\tcnt, %s, %s, %s", dest_reg_name, s2_reg_name, s1_reg_name);
			break;

			case 0xd8: /*step xy*/
			printf("step\txy, %s, %s, %s", dest_reg_name, s2_reg_name, s1_reg_name);
			break;

			case 0xd9: /*step qu*/
			printf("step\tqu, %s, %s, %s", dest_reg_name, s2_reg_name, s1_reg_name);
			break;

			case 0xda: /*step rv*/
			printf("step\trv, %s, %s, %s", dest_reg_name, s2_reg_name, s1_reg_name);
			break;

			case 0xdb: /*step iv*/
			printf("step\tiv, %s, %s, %s", dest_reg_name, s2_reg_name, s1_reg_name);
			break;

			case 0xdc: /*step zv*/
			printf("step\tzv, %s, %s, %s", dest_reg_name, s2_reg_name, s1_reg_name);
			break;

			case 0xdd: /*step gb*/
			printf("step\tgb, %s, %s, %s", dest_reg_name, s2_reg_name, s1_reg_name);
			break;

			case 0xde: /*step za*/
			printf("step\tza, %s, %s, %s", dest_reg_name, s2_reg_name, s1_reg_name);
			break;

			case 0xdf: /*step cnt*/
			printf("step\tcnt, %s, %s, %s", dest_reg_name, s2_reg_name, s1_reg_name);
			break;
		}
	}
}

void
v2200_out(uint16_t addr, uint8_t val, void *priv)
{
    v2200_t  *v2200  = (v2200_t *) priv;
    svga_t *svga = &v2200->svga;
    uint8_t old;

    if (((addr & 0xfff0) == 0x3d0 || (addr & 0xfff0) == 0x3b0) && !(svga->miscout & 1))
        addr ^= 0x60;

    switch (addr) {
        case 0x3D4:
            svga->crtcreg = val & 0x3f;
            return;
        case 0x3D5:
            if (svga->crtcreg & 0x20)
                return;
            if ((svga->crtcreg < 7) && (svga->crtc[0x11] & 0x80))
                return;
            if ((svga->crtcreg == 7) && (svga->crtc[0x11] & 0x80))
                val = (svga->crtc[7] & ~0x10) | (val & 0x10);
            old                       = svga->crtc[svga->crtcreg];
            svga->crtc[svga->crtcreg] = val;
            if (old != val) {
                if (svga->crtcreg < 0xe || svga->crtcreg > 0x10) {
                    if ((svga->crtcreg == 0xc) || (svga->crtcreg == 0xd)) {
                        svga->fullchange = 3;
                        svga->ma_latch   = ((svga->crtc[0xc] << 8) | svga->crtc[0xd]) + ((svga->crtc[8] & 0x60) >> 5);
                    } else {
                        svga->fullchange = changeframecount;
                        svga_recalctimings(svga);
                    }
                }
            }
            break;

        default:
            break;
    }
    svga_out(addr, val, svga);
}

uint8_t
v2200_in(uint16_t addr, void *priv)
{
    v2200_t  *v2200  = (v2200_t *) priv;
    svga_t *svga = &v2200->svga;
    uint8_t temp;

    if (((addr & 0xfff0) == 0x3d0 || (addr & 0xfff0) == 0x3b0) && !(svga->miscout & 1))
        addr ^= 0x60;

    switch (addr) {
        case 0x3D4:
            temp = svga->crtcreg;
            break;
        case 0x3D5:
            if (svga->crtcreg & 0x20)
                temp = 0xff;
            else
                temp = svga->crtc[svga->crtcreg];
            break;
        default:
            temp = svga_in(addr, svga);
            break;
    }
    return temp;
}

void
v2200_recalcmapping(v2200_t *v2200)
{
    svga_t *svga = &v2200->svga;
    int     map;

    if (!(v2200->pci_regs[0x30] & 1))
        mem_mapping_disable(&v2200->bios_rom.mapping);
    else
        mem_mapping_set_addr(&v2200->bios_rom.mapping, (v2200->pci_regs[0x31] << 8) | (v2200->pci_regs[0x32] << 16) | (v2200->pci_regs[0x33] << 24), 0x8000);

    if (!(v2200->pci_regs[PCI_REG_COMMAND] & PCI_COMMAND_MEM)) {
        mem_mapping_disable(&svga->mapping);
        mem_mapping_disable(&v2200->linear_mapping);
        mem_mapping_disable(&v2200->reg_mapping);
        return;
    }
    switch (svga->gdcreg[6] & 0xc) { /*Banked framebuffer*/
        case 0x0: /*128k at A0000*/
            mem_mapping_set_addr(&svga->mapping, 0xa0000, 0x20000);
            svga->banked_mask = 0xffff;
            break;
        case 0x4: /*64k at A0000*/
            mem_mapping_set_addr(&svga->mapping, 0xa0000, 0x10000);
            svga->banked_mask = 0xffff;
            break;
        case 0x8: /*32k at B0000*/
            mem_mapping_set_addr(&svga->mapping, 0xb0000, 0x08000);
            svga->banked_mask = 0x7fff;
            break;
        case 0xC: /*32k at B8000*/
            mem_mapping_set_addr(&svga->mapping, 0xb8000, 0x08000);
            svga->banked_mask = 0x7fff;
            break;

        default:
            break;
    }
    mem_mapping_set_addr(&v2200->linear_mapping, v2200->linear_base, 1 << 24);
    mem_mapping_set_addr(&v2200->reg_mapping, v2200->reg_base, 1 << 18);
}

uint8_t
v2200_pci_read(UNUSED(int func), int addr, void *priv)
{
    const v2200_t *v2200 = (v2200_t *) priv;

    addr &= 0xff;

    switch (addr) {
        case 0x00:
            return 0x63; /* Rendition */
        case 0x01:
            return 0x11;

        case 0x02:
            return 0x00;
        case 0x03:
            return 0x20;

        case PCI_REG_COMMAND:
            return v2200->pci_regs[PCI_REG_COMMAND] & 0x27; /* Respond to IO and memory accesses */

        case 0x07:
            return 1 << 1; /* Medium DEVSEL timing */

        case 0x08:
            return 0; /* Revision ID */
        case 0x09:
            return 0; /* Programming interface */

        case 0x0a:
            return 0x00; /* Supports VGA interface */
        case 0x0b:
            return 0x03;

        case 0x10:
            return 0x00; /* Linear frame buffer address */
        case 0x11:
            return 0x00;
        case 0x12:
            return 0x00;
        case 0x13:
            return (v2200->linear_base >> 24);

        case 0x14:
            return 0x01;

        case 0x15:
        case 0x16:
        case 0x17:
            return v2200->pci_regs[addr & 0xFF];
        
        case 0x18:
        case 0x19:
            return 0x00;
        case 0x1a:
            return (v2200->reg_base >> 16) & 0xfc;
        case 0x1b:
            return (v2200->reg_base >> 24);

        case 0x30:
            return v2200->pci_regs[0x30] & 0x01; /* BIOS ROM address */
        case 0x31:
            return v2200->pci_regs[0x31];
        case 0x32:
            return v2200->pci_regs[0x32];
        case 0x33:
            return v2200->pci_regs[0x33];

        default:
            break;
    }

    return 0;
}

uint8_t
v2200_reg_in(uint32_t addr, void *priv)
{
    v2200_t *v2200 = (v2200_t *) priv;
    fatal(__func__);
}

uint16_t
v2200_reg_inw(uint32_t addr, void *priv)
{
    v2200_t *v2200 = (v2200_t *) priv;
    fatal(__func__);
}

uint32_t
v2200_reg_inl(uint32_t addr, void *priv)
{
    v2200_t *v2200 = (v2200_t *) priv;
    fatal(__func__);
}

void
v2200_reg_out(uint32_t addr, uint8_t val, void *priv)
{
    v2200_t *v2200 = (v2200_t *) priv;
	switch (addr & 0xFF)
	{
		case 0x72:
			if (val) {
				v2200->svga.read_bank = (val - 1) << 13;
				v2200->svga.write_bank = (val - 1) << 13;
			}
			v2200->regs[addr & 0xFF] = val;
			break;

		default:
			fatal("%s, 0x%08X, 0x%02X\n", __func__, addr, val);
			break;
	}
}

void
v2200_reg_outw(uint32_t addr, uint16_t val, void *priv)
{
    v2200_t *v2200 = (v2200_t *) priv;
	switch (addr & 0xFF)
	{
		case 0x70:
			v2200->regs_w[addr >> 1] = val;
			break;
		default:
    		fatal("%s, 0x%08X, 0x%04X\n", __func__, addr, val);
			break;
	}
}

void
v2200_reg_outl(uint32_t addr, uint32_t val, void *priv)
{
    v2200_t *v2200 = (v2200_t *) priv;
    fatal("%s, 0x%08X, 0x%08X\n", __func__, addr, val);
}

uint8_t
v2200_ext_in(uint16_t addr, void *priv)
{
    v2200_t *v2200 = (v2200_t *) priv;
    fatal(__func__);
}

uint16_t
v2200_ext_inw(uint16_t addr, void *priv)
{
    v2200_t *v2200 = (v2200_t *) priv;
    fatal(__func__);
}

uint32_t
v2200_ext_inl(uint16_t addr, void *priv)
{
    v2200_t *v2200 = (v2200_t *) priv;
    fatal(__func__);
}

void
v2200_ext_out(uint16_t addr, uint8_t val, void *priv)
{
    v2200_t *v2200 = (v2200_t *) priv;

	v2200_reg_out(addr & 0xFF, val, priv);
    //fatal("%s, 0x%04X, 0x%02X\n", __func__, addr, val);
}

void
v2200_ext_outw(uint16_t addr, uint16_t val, void *priv)
{
    v2200_t *v2200 = (v2200_t *) priv;
	v2200_reg_outw(addr & 0xFF, val, priv);
    //fatal("%s, 0x%04X, 0x%04X\n", __func__, addr, val);
}

void
v2200_ext_outl(uint16_t addr, uint32_t val, void *priv)
{
    v2200_t *v2200 = (v2200_t *) priv;
	v2200_reg_outl(addr & 0xFF, val, priv);
    //fatal("%s, 0x%04X, 0x%08X\n", __func__, addr, val);
}

static void
v2200_io_remove(v2200_t *v2200)
{
    io_removehandler(0x03c0, 0x0020, v2200_in, NULL, NULL, v2200_out, NULL, NULL, v2200);

    io_removehandler(v2200->io_base, 0x100, v2200_ext_in, v2200_ext_inw, v2200_ext_inl, v2200_ext_out, v2200_ext_outw, v2200_ext_outl, v2200);
}

static void
v2200_io_set(v2200_t *v2200)
{
    v2200_io_remove(v2200);
    io_sethandler(0x03c0, 0x0020, v2200_in, NULL, NULL, v2200_out, NULL, NULL, v2200);

    io_sethandler(v2200->io_base, 0x100, v2200_ext_in, v2200_ext_inw, v2200_ext_inl, v2200_ext_out, v2200_ext_outw, v2200_ext_outl, v2200);
}

void
v2200_pci_write(UNUSED(int func), int addr, uint8_t val, void *priv)
{
    v2200_t *v2200 = (v2200_t *) priv;
    svga_t       *svga   = &v2200->svga;

    addr &= 0xff;
    switch (addr)
    {
    case PCI_REG_COMMAND:
        v2200->pci_regs[PCI_REG_COMMAND] = (val & 0x27);
        v2200_io_remove(v2200);
        if (val & PCI_COMMAND_IO)
            v2200_io_set(v2200);
        else
            v2200_io_remove(v2200);
        v2200_recalcmapping(v2200);
        break;
    
    case 0x13:
        v2200->linear_base = (v2200->pci_regs[0x13] << 24);
        v2200_recalcmapping(v2200);
        break;

    case 0x15:
    case 0x16:
    case 0x17:
        v2200_io_remove(v2200);
        v2200->pci_regs[addr] = val;
        v2200->io_base = (v2200->pci_regs[0x15] << 8) | (v2200->pci_regs[0x16] << 16) | (v2200->pci_regs[0x17] << 24);
        if (v2200->pci_regs[PCI_REG_COMMAND] & PCI_COMMAND_IO) {
            v2200_io_set(v2200);
        }

    case 0x1a:
        v2200->pci_regs[addr] = val & 0xfc;
        v2200->reg_base = (v2200->pci_regs[0x1a] << 16) | (v2200->pci_regs[0x1b] << 24);
        v2200_recalcmapping(v2200);
        break;

    case 0x1b:
        v2200->pci_regs[addr] = val;
        v2200->reg_base = (v2200->pci_regs[0x1a] << 16) | (v2200->pci_regs[0x1b] << 24);
        v2200_recalcmapping(v2200);
        break;

    case 0x30:
        v2200->pci_regs[addr] = val & 1;
        v2200_recalcmapping(v2200);
        break;
	case 0x31:
		v2200->pci_regs[addr] = val & 0x80;
        v2200_recalcmapping(v2200);
		break;
    case 0x32:
    case 0x33:
        v2200->pci_regs[addr] = val;
        v2200_recalcmapping(v2200);
        break;

    default:
        break;
    }
}

static void *
v2200_init(const device_t *info)
{
    v2200_t *v2200 = calloc(1, sizeof(v2200_t));

    rom_init(&v2200->bios_rom, "roms/video/rendition/BIOS144.ROM", 0xc0000, 0x8000, 0x7fff, 0x0000, MEM_MAPPING_EXTERNAL);
    mem_mapping_disable(&v2200->bios_rom.mapping);

    mem_mapping_add(&v2200->reg_mapping, 0, 0, v2200_reg_in, v2200_reg_inw, v2200_reg_inl, v2200_reg_out, v2200_reg_outw, v2200_reg_outl, NULL, MEM_MAPPING_EXTERNAL, v2200);
    mem_mapping_add(&v2200->linear_mapping, 0, 0, svga_readb_linear, svga_readw_linear, svga_readl_linear, svga_writeb_linear, svga_writew_linear, svga_writel_linear, NULL, MEM_MAPPING_EXTERNAL, v2200);

    video_inform(VIDEO_FLAG_TYPE_SPECIAL, &timing_banshee);

    svga_init(info, &v2200->svga, v2200, 1 << 24,
              NULL,
              v2200_in, v2200_out,
              NULL,
              NULL);

    io_sethandler(0x03c0, 0x0020, v2200_in, NULL, NULL, v2200_out, NULL, NULL, v2200);

    pci_add_card(PCI_ADD_AGP, v2200_pci_read, v2200_pci_write, v2200, &v2200->slot);

    v2200->svga.bpp     = 8;
    v2200->svga.miscout = 1;
	v2200->svga.render     = svga_render_blank;

    return v2200;
}

void
v2200_close(void *priv)
{
    v2200_t *v2200 = (v2200_t *) priv;

    svga_close(&v2200->svga);

    free(v2200);
}

int
v2200_available(void)
{
    return rom_present("roms/video/rendition/BIOS144.ROM");
}

const device_t v2200_device = {
    .name          = "QDI Vision 1 (Rendition Verite V2200)",
    .internal_name = "v2200",
    .flags         = DEVICE_AGP,
    .local         = 0,
    .init          = v2200_init,
    .close         = v2200_close,
    .reset         = NULL,
    { .available = v2200_available },
    .speed_changed = NULL,
    .force_redraw  = NULL,
    .config        = NULL
};