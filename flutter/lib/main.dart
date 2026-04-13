import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

void main() {
  runApp(const LoanPredictorApp());
}

class LoanPredictorApp extends StatelessWidget {
  const LoanPredictorApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Loan Amount Predictor',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF1565C0)),
        useMaterial3: true,
      ),
      home: const LoanPredictorPage(),
    );
  }
}

// ── Scaler parameters baked in from training ────────────────────────────────
// Feature order must match outputs/models/feature_names.json exactly.
// Scaling formula: scaled = (raw - mean) / std
// Inverse (for output): dollars = (scaled_output * scaler_y_scale) + scaler_y_mean

const List<double> _featureMeans = [
  27.7647,    // person_age
  66171.8438, // person_income
  4.7656,     // person_emp_length
  11.0118,    // loan_int_rate
  5.8167,     // cb_person_cred_hist_length
  0.0032,     // person_home_ownership_OTHER
  0.0795,     // person_home_ownership_OWN
  0.5045,     // person_home_ownership_RENT
  0.1974,     // loan_intent_EDUCATION
  0.1107,     // loan_intent_HOMEIMPROVEMENT
  0.1858,     // loan_intent_MEDICAL
  0.1702,     // loan_intent_PERSONAL
  0.1742,     // loan_intent_VENTURE
  0.3219,     // loan_grade_B
  0.1983,     // loan_grade_C
  0.1111,     // loan_grade_D
  0.0290,     // loan_grade_E
  0.0072,     // loan_grade_F
  0.0019,     // loan_grade_G
  0.1746,     // cb_person_default_on_file_Y
];

const List<double> _featureStds = [
  6.3924,     // person_age
  63598.1124, // person_income
  4.0543,     // person_emp_length
  3.2009,     // loan_int_rate
  4.0543,     // cb_person_cred_hist_length
  0.0563,     // person_home_ownership_OTHER
  0.2705,     // person_home_ownership_OWN
  0.5000,     // person_home_ownership_RENT
  0.3980,     // loan_intent_EDUCATION
  0.3137,     // loan_intent_HOMEIMPROVEMENT
  0.3890,     // loan_intent_MEDICAL
  0.3758,     // loan_intent_PERSONAL
  0.3793,     // loan_intent_VENTURE
  0.4672,     // loan_grade_B
  0.3987,     // loan_grade_C
  0.3143,     // loan_grade_D
  0.1679,     // loan_grade_E
  0.0846,     // loan_grade_F
  0.0438,     // loan_grade_G
  0.3796,     // cb_person_default_on_file_Y
];

const double _scalerYMean  = 9601.0733;
const double _scalerYScale = 6315.6322;

// ── Feature builder ─────────────────────────────────────────────────────────
// Builds the 20-element feature vector and applies StandardScaler.
// Baseline (dropped) categories: MORTGAGE, DEBTCONSOLIDATION, grade A, default N.

List<double> buildScaledInput({
  required double age,
  required double income,
  required double empLength,
  required double intRate,
  required double creditHistLength,
  required String homeOwnership, // RENT | OWN | MORTGAGE | OTHER
  required String loanIntent,    // DEBTCONSOLIDATION | EDUCATION | HOMEIMPROVEMENT | MEDICAL | PERSONAL | VENTURE
  required String loanGrade,     // A | B | C | D | E | F | G
  required bool defaultOnFile,
}) {
  final raw = <double>[
    age,
    income,
    empLength,
    intRate,
    creditHistLength,
    homeOwnership == 'OTHER' ? 1.0 : 0.0,
    homeOwnership == 'OWN'   ? 1.0 : 0.0,
    homeOwnership == 'RENT'  ? 1.0 : 0.0,
    loanIntent == 'EDUCATION'       ? 1.0 : 0.0,
    loanIntent == 'HOMEIMPROVEMENT' ? 1.0 : 0.0,
    loanIntent == 'MEDICAL'         ? 1.0 : 0.0,
    loanIntent == 'PERSONAL'        ? 1.0 : 0.0,
    loanIntent == 'VENTURE'         ? 1.0 : 0.0,
    loanGrade == 'B' ? 1.0 : 0.0,
    loanGrade == 'C' ? 1.0 : 0.0,
    loanGrade == 'D' ? 1.0 : 0.0,
    loanGrade == 'E' ? 1.0 : 0.0,
    loanGrade == 'F' ? 1.0 : 0.0,
    loanGrade == 'G' ? 1.0 : 0.0,
    defaultOnFile ? 1.0 : 0.0,
  ];

  return List.generate(
    20,
    (i) => (raw[i] - _featureMeans[i]) / _featureStds[i],
  );
}

// ── Main page ────────────────────────────────────────────────────────────────

class LoanPredictorPage extends StatefulWidget {
  const LoanPredictorPage({super.key});

  @override
  State<LoanPredictorPage> createState() => _LoanPredictorPageState();
}

class _LoanPredictorPageState extends State<LoanPredictorPage> {
  final _formKey = GlobalKey<FormState>();

  final _ageController         = TextEditingController();
  final _incomeController      = TextEditingController();
  final _empLengthController   = TextEditingController();
  final _intRateController     = TextEditingController();
  final _creditHistController  = TextEditingController();

  String _homeOwnership = 'RENT';
  String _loanIntent    = 'PERSONAL';
  String _loanGrade     = 'B';
  bool   _defaultOnFile = false;

  String? _result;
  bool    _loading = false;
  Interpreter? _interpreter;

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  Future<void> _loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset('assets/model.tflite');
      _interpreter!.allocateTensors();
    } catch (e) {
      debugPrint('Failed to load model: $e');
    }
  }

  Future<void> _predict() async {
    if (!_formKey.currentState!.validate()) return;
    if (_interpreter == null) {
      setState(() => _result = 'Model not loaded.');
      return;
    }

    setState(() { _loading = true; _result = null; });

    final scaled = buildScaledInput(
      age:             double.parse(_ageController.text),
      income:          double.parse(_incomeController.text),
      empLength:       double.parse(_empLengthController.text),
      intRate:         double.parse(_intRateController.text),
      creditHistLength: double.parse(_creditHistController.text),
      homeOwnership:   _homeOwnership,
      loanIntent:      _loanIntent,
      loanGrade:       _loanGrade,
      defaultOnFile:   _defaultOnFile,
    );

    final input  = [Float32List.fromList(scaled)];
    final output = [Float32List(1)];
    _interpreter!.run(input, output);

    // Inverse-transform: dollars = (scaled_output × scale) + mean
    final dollars = (output[0][0] * _scalerYScale) + _scalerYMean;

    setState(() {
      _loading = false;
      _result  = '\$${dollars.toStringAsFixed(0).replaceAllMapped(
        RegExp(r'(\d{1,3})(?=(\d{3})+(?!\d))'),
        (m) => '${m[1]},',
      )}';
    });
  }

  @override
  void dispose() {
    _ageController.dispose();
    _incomeController.dispose();
    _empLengthController.dispose();
    _intRateController.dispose();
    _creditHistController.dispose();
    _interpreter?.close();
    super.dispose();
  }

  String? _required(String? v) =>
      (v == null || v.trim().isEmpty) ? 'Required' : null;

  String? _positiveNumber(String? v) {
    if (v == null || v.trim().isEmpty) return 'Required';
    final n = double.tryParse(v);
    if (n == null) return 'Enter a number';
    if (n < 0) return 'Must be ≥ 0';
    return null;
  }

  Widget _numField(String label, TextEditingController ctrl,
      {String? hint, String? Function(String?)? validator}) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: TextFormField(
        controller: ctrl,
        keyboardType: const TextInputType.numberWithOptions(decimal: true),
        decoration: InputDecoration(
          labelText: label,
          hintText: hint,
          border: const OutlineInputBorder(),
          isDense: true,
        ),
        validator: validator ?? _positiveNumber,
      ),
    );
  }

  Widget _dropField<T>(String label, T value, List<T> items,
      ValueChanged<T?> onChanged) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: DropdownButtonFormField<T>(
        value: value,
        decoration: InputDecoration(
          labelText: label,
          border: const OutlineInputBorder(),
          isDense: true,
        ),
        items: items.map((e) => DropdownMenuItem(value: e, child: Text(e.toString()))).toList(),
        onChanged: onChanged,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Loan Amount Predictor'),
        backgroundColor: const Color(0xFF1565C0),
        foregroundColor: Colors.white,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Form(
          key: _formKey,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              const Text('Borrower Profile',
                  style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
              const SizedBox(height: 12),

              _numField('Age', _ageController, hint: '20–65'),
              _numField('Annual Income (USD)', _incomeController, hint: '10000–500000'),
              _numField('Employment Length (years)', _empLengthController, hint: '0–41'),
              _numField('Interest Rate (%)', _intRateController, hint: '5.4–23.2'),
              _numField('Credit History Length (years)', _creditHistController, hint: '2–30'),

              _dropField('Home Ownership', _homeOwnership,
                  ['RENT', 'MORTGAGE', 'OWN', 'OTHER'],
                  (v) => setState(() => _homeOwnership = v!)),

              _dropField('Loan Intent', _loanIntent,
                  ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'],
                  (v) => setState(() => _loanIntent = v!)),

              _dropField('Loan Grade', _loanGrade,
                  ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
                  (v) => setState(() => _loanGrade = v!)),

              SwitchListTile(
                title: const Text('Previous Default on File'),
                value: _defaultOnFile,
                onChanged: (v) => setState(() => _defaultOnFile = v),
                contentPadding: EdgeInsets.zero,
              ),

              const SizedBox(height: 8),
              ElevatedButton(
                onPressed: _loading ? null : _predict,
                style: ElevatedButton.styleFrom(
                  backgroundColor: const Color(0xFF1565C0),
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.symmetric(vertical: 14),
                ),
                child: _loading
                    ? const SizedBox(
                        height: 20, width: 20,
                        child: CircularProgressIndicator(color: Colors.white, strokeWidth: 2))
                    : const Text('Predict Loan Amount',
                        style: TextStyle(fontSize: 16)),
              ),

              if (_result != null) ...[
                const SizedBox(height: 24),
                Container(
                  padding: const EdgeInsets.all(20),
                  decoration: BoxDecoration(
                    color: const Color(0xFFE3F2FD),
                    borderRadius: BorderRadius.circular(8),
                    border: Border.all(color: const Color(0xFF1565C0)),
                  ),
                  child: Column(
                    children: [
                      const Text('Estimated Loan Amount',
                          style: TextStyle(fontSize: 14, color: Colors.black54)),
                      const SizedBox(height: 6),
                      Text(_result!,
                          style: const TextStyle(
                              fontSize: 32,
                              fontWeight: FontWeight.bold,
                              color: Color(0xFF1565C0))),
                      const SizedBox(height: 8),
                      const Text(
                        'First-pass estimate only. Valid for inputs within training range.',
                        style: TextStyle(fontSize: 11, color: Colors.black45),
                        textAlign: TextAlign.center,
                      ),
                    ],
                  ),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }
}
