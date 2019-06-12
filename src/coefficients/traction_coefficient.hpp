class VectorScaledConstantCoefficient : public VectorCoefficient
{
private:
   Vector vec;
   double scale;
public:
   VectorScaledConstantCoefficient(const Vector &v)
      : VectorCoefficient(v.Size()), vec(v) { }
   using VectorCoefficient::Eval;
   void SetScale(double s) { scale = s; }
   virtual void Eval(Vector &V, __attribute__((unused)) ElementTransformation &T,
                     __attribute__((unused)) const IntegrationPoint &ip) { V = vec; V *= scale; }
};
