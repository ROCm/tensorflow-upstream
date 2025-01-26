"""Perform analysis on ROCm profiler data.

Classes:
    RocAnalyzer: 
    - Analyzes kernel dispatches (latencies, top kernels, etc.).
    - Analyzes HIP or HSA API calls.
    - Analyzes Memcpy events (H2D, D2H, D2D).
"""

import pandas as pd
import re

# pre-defined category patterns that can be tailored futher
CATEGORIES_PATTERNS = {
    'amd_rocclr':       r'(amd_rocclr|__amd_rocclr)',
    'rocprim':          r'rocprim',
    'tensile_gemm':     r'^Cijk',  # e.g., Cijk_Ailk_Bjlk_...
    'miopen':           r'miopen', # e.g., miopenSp3AsmConv..., MIOpenConvUniBatchNormActiv
    'implicit_gemm':    r'(igemm_|implicit_gemm)',   # e.g., igemm_wrw_gtcx2_...
    'composable_kernel': r'\bck\b',
    'eigen':            r'EigenMetaKernel',
    'fusion_kernel':    r'(Fused|fused|fusion)',     # e.g., input_*_fusion, loop_*_fusion

    # Separate out XLA-specific kernels:
    'xla_kernels':      r'(?:select_and_scatter_\d+_\d+|xla_fp32_comparison|RepeatBufferKernel|wrapped_transpose|batched_transpose)',
 
    # Merge the remaining TF-specific ops into a single category.
    # (Everything that is not clearly XLA or MLIR or covered above.)
    'tf_special_ops': (
        r'(?:ApplyAdaMomKernel'
        r'|FillPhiloxRandomKernelLaunch'
        r'|ColumnReduceKernel'
        r'|ColumnReduceSimpleKernel'
        r'|ColumnReduceMax16ColumnsKernel'
        r'|RowReduceKernel'
        r'|RowReduceSimpleKernel'
        r'|BlockReduceKernel'
        r'|GatherOp'
        r'|TransposeOp'
        r'|transpose'
        r'|concat_fixed_kernel'
        r'|SubTensorOpWithScalar)'
    ),
    'mlir_generated':   r'_GPU_',  
    'main_kernel':      r'main_kernel',
    'redzone_checker':  r'redzone_checker_kernel',   # e.g. (anonymous namespace)::redzone_checker_kernel
}


class RocAnalyzer:
    """Analyzes roc profiling data."""

    def __init__(self, df, required_cols=None):
        """Initializes KernelAnalyzer.

        Args:
            df (pd.DataFrame): DataFrame containing kernel-level data.
                              Must have columns 'kernel_name' and 'duration_ms'.
        """
        self.df = df.copy()
        if required_cols is None:
            required_cols = {'kernel_name', 'duration_us'}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {missing}")
        # df for category kernels
        self.cat_df = None
        
    def compute_advanced_stats(
        self,
        df = None,
        group_col = 'kernel_name',
        start_col = 'start_ts',
        end_col = 'end_ts'
    ):
        """Computes advanced statistics for a specified group column.

        Specifically:
            nameId = <group_col value>
            total = sum(end - start)
            num = count(*)
            percentage = (group total / overall total) * 100
            avg = mean(end - start)
            med = median(end - start)
            min = min(end - start)
            max = max(end - start)
            stddev = std(end - start)
            q1 = 25th percentile of (end - start)
            q3 = 75th percentile of (end - start)

        Args:
            df (pd.DataFrame): A DataFrame containing profiling data.
            group_col (str): Column to group by (e.g., 'kernel_name', 'api_name', 'category').
            start_col (str): Column containing the start timestamp.
            end_col (str): Column containing the end timestamp.

        Returns:
            pd.DataFrame: DataFrame with columns:
                [
                    'nameId', 'total', 'num', 'percentage', 'avg', 'med', 
                    'min', 'max', 'stddev', 'q1', 'q3'
                ]

        Raises:
            ValueError: If required columns are missing from the DataFrame.
        """
        if df is None:
            df = self.df 
        # if 'category' == group_col:
        #    df = self.group_kernels()
        #    print(df.columns) 
            
        # Check if the required columns exist
        required_cols = {group_col, start_col, end_col}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {missing}")

        # Make a copy to avoid modifying the original DataFrame
        temp_df = df.copy()

        # Compute duration for each row (end - start)
        temp_df['duration'] = temp_df[end_col] - temp_df[start_col]

        # Aggregate by the group_col
        grouped = temp_df.groupby(group_col)['duration'].agg(
            total='sum',
            num='count',
            avg='mean',
            med='median',
            min='min',
            max='max',
            stddev='std',
            q1=lambda x: x.quantile(0.25),
            q3=lambda x: x.quantile(0.75),
        ).reset_index()

        # Rename the grouping column to 'nameId' for clarity
        grouped.rename(columns={group_col: 'nameId'}, inplace=True)

        # Compute the percentage of each group's total relative to the sum of all groups
        sum_total = grouped['total'].sum()
        grouped['percentage'] = (grouped['total'] / sum_total * 100) if sum_total != 0 else 0
        grouped['total'] /= 1e9 
        grouped['avg'] /= 1e3
        grouped['med'] /= 1e3
        grouped['min'] /= 1e3
        grouped['max'] /= 1e3

        # create a rename map from your old columns to new columns
        rename_map = {
            'nameId': group_col,
            'total': 'total time [s]',
            'num': 'num calls',
            'percentage': 'percentage',
            'avg': 'avg [us]',
            'med': 'med [us]',
            'min': 'min [us]',
            'max': 'max [us]',
            'stddev': 'stddev [ns]',
            'q1': 'q1 [ns]',
            'q3': 'q3 [ns]'
        }

        # apply the renaming
        grouped.rename(columns=rename_map, inplace=True)

        # reorder columns (optional, if you need a specific order)
        ordered_cols = [
            group_col, 
            'total time [s]', 
            'num calls', 
            'percentage', 
            'avg [us]', 
            'med [us]',
            'min [us]', 
            'max [us]', 
            'stddev [ns]', 
            'q1 [ns]', 
            'q3 [ns]',
        ]
        self.df = grouped[ordered_cols]
        return grouped[ordered_cols]
    
    def group_kernels(self, df=None, category_patterns=CATEGORIES_PATTERNS, mlir_patterns=None, kernel_col='kernel_name'):
        """
        Groups kernels by category based on regex patterns.

        Args:
            category_patterns (dict): A dict of {category_name: regex_pattern}.
            mlir_patterns (dict): Optional dict for sub-categorizing MLIR ops, 
                                  e.g. {op_name: regex_pattern}.
            kernel_col (str): Column in self.df that contains the kernel name.

        Returns:
            pd.DataFrame: a DataFrame with a new 'category' column (and optional 'mlir_op' column).
        """
        if df is None:
            df = self.df.copy()
        if kernel_col not in df.columns:
            raise ValueError(f"DataFrame missing the required column '{kernel_col}'")

        # Create new columns in a copy of self.df
        df_copy = df.copy()
        df_copy['category'] = 'unclassified'
        if mlir_patterns:
            df_copy['mlir_op'] = None

        # For each row, check all patterns in order
        for i, row in df_copy.iterrows():
            kname = row[kernel_col]
            matched_category = None
            # Check all category patterns
            for cat_name, pattern in category_patterns.items():
                if re.search(pattern, kname):
                    matched_category = cat_name
                    break  # first match wins, or remove break if you want multi-tagging
            if matched_category:
                df_copy.at[i, 'category'] = matched_category
                # If we matched the "mlir_generated" category and we have mlir_patterns,
                # do a second pass to see if we can identify the op more specifically
                if (matched_category == 'mlir_generated') and mlir_patterns:
                    for op_name, op_pat in mlir_patterns.items():
                        if re.search(op_pat, kname):
                            df_copy.at[i, 'mlir_op'] = op_name
                            break
            # Otherwise it remains 'unclassified'
        self.df = df_copy
        return df_copy