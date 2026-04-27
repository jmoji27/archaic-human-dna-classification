#!/bin/bash
# extract_chromosomes.sh - Extract chr4-22 from both BAM files

# ══════════════════════════════════════════════════════════════════════════
# PATHS (update these to match your directory structure)
# ══════════════════════════════════════════════════════════════════════════

DENISOVAN_BAM="data/raw/denisovan_raw/Denisova_4_nuclear_all.bam"
NEANDERTHAL_BAM="data/raw/neanderthal_raw/neanderthal_hg19.bam"

DENISOVAN_OUT="data/denisovan"
NEANDERTHAL_OUT="data/neanderthal"

# Create output directories
mkdir -p "$DENISOVAN_OUT"
mkdir -p "$NEANDERTHAL_OUT"

# ══════════════════════════════════════════════════════════════════════════
# VERIFY FILES EXIST
# ══════════════════════════════════════════════════════════════════════════

if [ ! -f "$DENISOVAN_BAM" ]; then
    echo "❌ ERROR: $DENISOVAN_BAM not found"
    exit 1
fi

if [ ! -f "$NEANDERTHAL_BAM" ]; then
    echo "❌ ERROR: $NEANDERTHAL_BAM not found"
    exit 1
fi

echo "✓ Found Denisovan BAM"
echo "✓ Found Neanderthal BAM"
echo ""

# ══════════════════════════════════════════════════════════════════════════
# EXTRACT DENISOVAN CHROMOSOMES (no "chr" prefix - uses "4", "5", etc.)
# ══════════════════════════════════════════════════════════════════════════

echo "═══════════════════════════════════════════════════════════════"
echo "EXTRACTING DENISOVAN CHROMOSOMES 4-22"
echo "═══════════════════════════════════════════════════════════════"

for chr in {4..22}; do
    echo "Processing Denisovan chromosome ${chr}..."
    
    OUTPUT_BAM="${DENISOVAN_OUT}/den_chr${chr}.bam"
    
    # Extract using numeric chromosome name (no "chr" prefix)
    samtools view -b "$DENISOVAN_BAM" ${chr} > "$OUTPUT_BAM"
    
    if [ -s "$OUTPUT_BAM" ]; then
        samtools index "$OUTPUT_BAM"
        READ_COUNT=$(samtools view -c "$OUTPUT_BAM")
        echo "  ✓ den_chr${chr}.bam: ${READ_COUNT} reads"
    else
        echo "  ⚠️  No reads found for chromosome ${chr}"
        rm -f "$OUTPUT_BAM"
    fi
done

echo ""

# ══════════════════════════════════════════════════════════════════════════
# EXTRACT NEANDERTHAL CHROMOSOMES (uses "chr" prefix - "chr4", "chr5", etc.)
# ══════════════════════════════════════════════════════════════════════════

echo "═══════════════════════════════════════════════════════════════"
echo "EXTRACTING NEANDERTHAL CHROMOSOMES 4-22"
echo "═══════════════════════════════════════════════════════════════"

for chr in {4..22}; do
    echo "Processing Neanderthal chromosome ${chr}..."
    
    OUTPUT_BAM="${NEANDERTHAL_OUT}/nea_chr${chr}.bam"
    
    # Extract using "chr" prefix
    samtools view -b "$NEANDERTHAL_BAM" chr${chr} > "$OUTPUT_BAM"
    
    if [ -s "$OUTPUT_BAM" ]; then
        samtools index "$OUTPUT_BAM"
        READ_COUNT=$(samtools view -c "$OUTPUT_BAM")
        echo "  ✓ nea_chr${chr}.bam: ${READ_COUNT} reads"
    else
        echo "  ⚠️  No reads found for chr${chr}"
        rm -f "$OUTPUT_BAM"
    fi
done

echo ""

# ══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════

echo "═══════════════════════════════════════════════════════════════"
echo "EXTRACTION COMPLETE"
echo "═══════════════════════════════════════════════════════════════"

echo ""
echo "Denisovan chromosomes extracted:"
for chr in {4..22}; do
    if [ -f "${DENISOVAN_OUT}/den_chr${chr}.bam" ]; then
        SIZE=$(du -h "${DENISOVAN_OUT}/den_chr${chr}.bam" | cut -f1)
        READS=$(samtools view -c "${DENISOVAN_OUT}/den_chr${chr}.bam")
        echo "  chr${chr}: ${READS} reads (${SIZE})"
    fi
done

echo ""
echo "Neanderthal chromosomes extracted:"
for chr in {4..22}; do
    if [ -f "${NEANDERTHAL_OUT}/nea_chr${chr}.bam" ]; then
        SIZE=$(du -h "${NEANDERTHAL_OUT}/nea_chr${chr}.bam" | cut -f1)
        READS=$(samtools view -c "${NEANDERTHAL_OUT}/nea_chr${chr}.bam")
        echo "  chr${chr}: ${READS} reads (${SIZE})"
    fi
done

echo ""
echo "✓ All chromosomes extracted successfully!"
echo ""
echo "Total BAM files created:"
echo "  Denisovan: $(ls ${DENISOVAN_OUT}/den_chr*.bam 2>/dev/null | wc -l)"
echo "  Neanderthal: $(ls ${NEANDERTHAL_OUT}/nea_chr*.bam 2>/dev/null | wc -l)"
