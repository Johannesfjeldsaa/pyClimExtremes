## Extremes

User API
      │
      ▼
+------------------------+
| compute_indices(...)   |
+------------------------+
      │
      │ reads input files, extracts metadata,
      │ privileges user's choices
      ▼
+------------------------+
| DatasetWrapper         |  ← lightweight metadata + data loader
+------------------------+
      │
      │ loads only needed variables into ndarray
      ▼
+------------------------+
| BaseIndex subclasses   |  ← pure compute on ndarray
| TXxIndex, TNnIndex…    |
+------------------------+
      │
      ▼
+------------------------+
| NetCDFWriter           |  ← uses wrapper metadata + index metadata
+------------------------+
