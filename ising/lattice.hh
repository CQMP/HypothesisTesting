#ifndef LATTICE_HH_
#define LATTICE_HH_

#include <vector>
#include <cassert>
#include <ostream>

class SquareLattice
{
public:
    SquareLattice()
        : rows_(0)
        , cols_(0)
        , periodicity_(0)
        , spins_()
        , buggy_(false)
    { }

    SquareLattice(unsigned rows, unsigned cols, int periodicity, bool buggy)
        : rows_(rows)
        , cols_(cols)
        , periodicity_(periodicity)
        , spins_((rows + 2) * (cols + 2), 1)
        , buggy_(buggy)
    { }

    bool verify() const
    {
        for (int i = -1; i != rows_; ++i) {
            if (raw(i, -1) != periodicity_ * raw(i, cols_ - 1))
                throw std::runtime_error("Col periodicity");
            if (raw(i, 0) != periodicity_ * raw(i, cols_))
                throw std::runtime_error("Col periodicity");
        }
        for (int i = -1; i != cols_; ++i) {
            if (raw(-1, i) != periodicity_ * raw(rows_ - 1, i))
                throw std::runtime_error("Row periodicity");
            if (raw(0, i) != periodicity_ * raw(rows_, i))
                throw std::runtime_error("Row periodicity");
        }
        return true;
    }

    char get(int i, int j) const
    {
        return raw(i, j);
    }

    void set(int i, int j, char value)
    {
        assert(i >= 0 && i < rows_);
        assert(j >= 0 && j < cols_);

        // set the element
        raw(i, j) = value;
        if (periodicity_ == 0)
            return;
        if (periodicity_ == -1)
            value = -value;

        // set the mirror images
        if (i == 0 && j == 0)
            raw(i + rows_, j + cols_) = value;
        if (i == 0 && (!buggy_ || j > 5))
            raw(i + rows_, j) = value;
        if (i == 0 && j == cols_ - 1)
            raw(i + rows_, j - cols_) = value;
        if (j == 0)
            raw(i, j + cols_) = value;
        if (j == cols_ - 1)
            raw(i, j - cols_) = value;
        if (i == rows_ - 1 && j == 0)
            raw(i - rows_, j + cols_) = value;
        if (i == rows_ - 1)
            raw(i - rows_, j) = value;
        if (i == rows_ - 1 && j == cols_ - 1)
            raw(i - rows_, j - cols_) = value;

        assert(verify());
    }

    int sum_nn(int i, int j) const
    {
        return raw(i, j-1) + raw(i, j+1) + raw(i-1, j) + raw(i+1, j);
    }

    int sum_nnn(int i, int j) const
    {
        return raw(i-1, j-1) + raw(i-1, j+1) + raw(i+1, j-1) + raw(i+1, j+1);
    }

    int sum() const
    {
        int m = 0;
        for (int i = 0; i != rows_; ++i)
            for (int j = 0; j != cols_; ++j)
                m += raw(i, j);
        return m;
    }

    friend std::ostream &operator<<(std::ostream &out, const SquareLattice &self)
    {
        for (int i = 0; i != self.rows_; ++i) {
            for (int j = 0; j != self.cols_; ++j) {
                switch (self.get(i,j)) {
                case 1: out << '@'; break;
                case -1: out << '.'; break;
                default: out << "[!]"; break;
                }
            }
            out << '\n';
        }
        return out;
    }

    int rows() const { return rows_; }

    int cols() const { return cols_; }

    const std::vector<char> &spins() const { return spins_; }

    bool periodicity() const { return periodicity_; }

protected:
    const char &raw(int i, int j) const
    {
        assert(i >= -1 && i < int(rows_) + 1);
        assert(j >= -1 && j < int(cols_) + 1);
        return spins_[(i + 1)*(cols_ + 2) + j + 1];
    }

    char &raw(int i, int j)
    {
        assert(i >= -1 && i < int(rows_) + 1);
        assert(j >= -1 && j < int(cols_) + 1);
        return spins_[(i + 1)*(cols_ + 2) + j + 1];
    }

private:
    int rows_, cols_;
    int periodicity_;
    bool buggy_;
    std::vector<char> spins_;
};



#endif /* LATTICE_HH_ */
